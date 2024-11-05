# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 08:44:56 2024

@author: Xin
"""
#Part 2
##step two 定义训练mobilenet_V2模型(模型已定义只需训练)
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from model_v2 import MobileNetV2
import torchvision.models.mobilenetv2
from PIL import Image
import matplotlib.pyplot as plt
 

##训练模型 
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
 
    batch_size = 16
    epochs = 20
 
    data_transform = {
        "train": transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
 
    data_root = 'd:\\'  # get data root path
    image_path = os.path.join(data_root, "data_set")  # data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
 
    # {'TUM':1, 'NORM':0}
    class_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in class_list.items())
    # write dict into json file
    json_Bstr = json.dumps(cla_dict, indent=4)
    with open('class_B_list.json', 'w') as json_file:
        json_file.write(json_Bstr)
 
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
 
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
 
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "validation"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
 
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
 
    # create model
    net = MobileNetV2(num_classes=2)
 
    # load pretrain weights
    # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    model_weight_path = "d:\\code\\mobilenet_v2.pth"
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location='cpu') 
 
    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()} #字典类型
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)
 
    # freeze features weights
    for param in net.features.parameters():
        param.requires_grad = False
 
    net.to(device)
 
    # define loss function
    loss_function = nn.CrossEntropyLoss()
 
    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)
 
    best_acc = 0.0
    save_path = 'd:\\code\\MobileNetV2.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
 
            # print statistics
            running_loss += loss.item()
 
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
 
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
 
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
 
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
 
    print('Finished Training')
 
 
#if __name__ == '__main__':
    main()

##step three 将训练好的模型用于组织区分（将Part1生成的mucosa_tile_index生成的tiles带入模型区分出肿瘤tile及得到相应坐标）
def main(img_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
 
    # load image
 
    # read class_indict
    json_path = 'd:\\data\\class_B_list.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
 
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    seq=1  ##全部预测tile 为tum的概率
    # create model
    model = MobileNetV2(num_classes=2).to(device)
    # load model weights
    model_weight_path = "d:\\code\\MobileNetV2.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
            img = Image.open(img_path)
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            tumor_class=class_indict[str(predict_cla)]
            prob = predict[seq].numpy()
    return  tumor_class, prob

##Example
slide_num=1
img_png_dir=os.listdir('d:\\mucosa_tile_png\\'+str(slide_num).zfill(4)+'\\')
Tumor_class_list=[]
for i in range(0,len(img_png_dir)):
    img_png_path=os.path.join('d:\\mucosa_tile_png\\'+str(slide_num).zfill(4)+'\\',img_png_dir[i])
    Tumor_class=main(img_png_path)[0]
    Tumor_class_list.append(Tumor_class)

index_list=[]    
for index,string in enumerate(Tumor_class_list):
    if 'TUM' in string:
        index_list.append(index)
        
##获取final_tile_index
mucosa_tile_index=np.load(os.path.join('d:\\mucosa_tile_index',"{}.npy".format(str(slide_num).zfill(4))) )
final_tile_index=[]
for seq in index_list:
  cancer_tile_index = mucosa_tile_index[seq]
  final_tile_index.append(cancer_tile_index)
  
tile_index_filename = os.path.join('d:\\final_tile_index',"{}.npy".format(str(slide_num).zfill(4)))
np.save(tile_index_filename, final_tile_index)  ##保存WSI相对应肿瘤组织的坐标
#tile_index_num=np.load(os.path.join('d:\\final_tile_index',"{}.npy".format(str(slide_num).zfill(4))) ###读取坐标  


#获取每张tile预测prob概率

##path_img='c:\\data_set\\test'
def get_prob_list(path_img):
    img_dir=os.listdir(path_img)
    prob_list=[]
    for i in img_dir:
        img_path=os.path.join(path_img,i)
        prob=main(img_path)[1]
        prob_list.append(prob)
    return prob_list

