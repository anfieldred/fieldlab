# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:18:03 2024

@author: Xin
"""

#Part 3 get mean prob
##step two 定义及训练moblienet V3模型
import os
import math
import sys
import json
import pandas as pd 
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from mobilenet_v3 import mobilenet_v3_large

##训练模型
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
 
    batch_size = 16
    epochs = 60
 
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
 
    data_root = 'd:\\'  # get data root path
    image_path = os.path.join(data_root, "data_sets")  # data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
 
    # {'N0':0, 'N1':1, 'N2':2, 'N3a':3, 'N3b':4}   {'N':0, 'P':1}
    N_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in N_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
 
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
    net = mobilenet_v3_large(num_classes=2)  ##{'N0':0, 'N1':1, 'N2':2, 'N3a':3, 'N3b':4} num_classes=5
 
    # load pretrain weights
    # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    model_weight_path = "d:\\code\\mobilenet_v3_large.pth"
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
    optimizer = optim.Adam(params, lr=0.000001)
 
    best_acc = 0.0
    save_path = 'd:\\code\\MobileNetV3.pth'
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

##step three 获取每张WSI的平均prob
N_dict = {'N':0,'P':1}
N_status_dataframe=pd.read_excel('d:\\data\\N_p_n.xlsx')
N_status_array=N_status_dataframe.values

def main(img_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
 
    # load image
 
    # read class_indict
    json_path = 'd:\\data\\class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
 
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    ##seq=N_dict[N_status_array[slide_num-1][1]] ##N_status_array[slide_num-1][1]为WSI N label标签 
    seq=1   ##全部预测N positive的概率  这样N标记阴性的tile获得阳性的概率值低 N标记阳性的tile获得阳性的概率值高
    # create model
    model = mobilenet_v3_large(num_classes=2).to(device)
    # load model weights
    model_weight_path = "d:\\code\\MobileNetV3.pth"
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
            N_class=class_indict[str(predict_cla)]
            prob = predict[seq].numpy()  ##WSI上N真实分期对应的prob，非sample预测N分期所得prob；N_status_array[slide_num-1][1]为WSI N标签 
    return  N_class, prob   ##seq=N_dict[N_status_array[slide_num-1][1]]

N_status_dataframe=pd.read_excel('d:\\data\\N_p_n.xlsx')
N_status_array=N_status_dataframe.values
#get tile numbers of each slide_num 
def get_tiles_num(slide_num):    
  if slide_num in range(1,1444):
      base_file='d:\\data_sets\\train\\'
  elif slide_num in range(1444,1819):
      base_file='d:\\data_sets\\validation\\'
  else: base_file='d:\\data_sets\\test\\' 
  img_png_dir=os.path.join(base_file+str(N_status_array[slide_num-1][1])+'\\')
  files=os.listdir(img_png_dir)
  tiles_name_list=[]
  for i in files:
      if str(slide_num).zfill(4)+'_' in i:
          tiles_name = i
          tiles_name_list.append(tiles_name)
      len_tiles=len(tiles_name_list)
  return len_tiles

def get_mean_prob(slide_num):
  if slide_num in range(1,1444):
      base_file='d:\\data_sets\\train\\'
  elif slide_num in range(1444,1819):
      base_file='d:\\data_sets\\validation\\'
  else: base_file='d:\\data_sets\\test\\'   
  img_png_dir=os.path.join(base_file+str(N_status_array[slide_num-1][1])+'\\')
  prob_list=[]
  for i in range(0,get_tiles_num(slide_num)):
    img_png_path=os.path.join(img_png_dir + str(slide_num).zfill(4)+'_'+str(i).zfill(5) + '.png')
    prob=main(img_png_path)[1]
    prob_list.append(prob)
  if prob_list==[]:   ##WSI图像差slide_num对应WSI无可用tiles生成，防止报错nan
      mean_prob=str(0)
  else: 
      mean_prob=np.mean(prob_list,where=(~np.isnan(prob_list)))
      mean_prob=round(mean_prob,2)
  return mean_prob

def get_mean_prob_list(starting,ending):
  mean_prob_list=[]
  for slide_num in range(starting,ending):
    mean_prob=get_mean_prob(slide_num)
    mean_prob_list.append(mean_prob)
  return mean_prob_list



#mean_prob_list_dataframe=pd.DataFrame(mean_prob_list)
#mean_prob_list_dataframe.to_excel('d:\\mean_prob_list_dataframe.xlsx',index=False) 

