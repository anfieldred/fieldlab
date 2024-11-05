# -*- coding: utf-8 -*-
"""
Created on Sun May 19 18:52:50 2024

@author: Xin
"""

#TCGA Part 2
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
import math
import os
import re
import numpy as np
import openslide
import PIL
from PIL import Image
from openslide import OpenSlideError
from openslide.deepzoom import DeepZoomGenerator
import pandas as pd
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
folder="d:\\TCGA"
SCALE_FACTOR=32    
img_tile_size=16     #16*32=512(tile size)     32*32=1024
starting=1
ending=248

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
    return  tumor_class

img_png_dir=os.listdir('d:\\TCGA_mucosa_tile_png\\'+str(slide_num).zfill(4)+'\\')
Tumor_class_list=[]
for i in range(0,len(img_png_dir)):
    img_png_path=os.path.join('d:\\TCGA_mucosa_tile_png\\'+str(slide_num).zfill(4)+'\\',img_png_dir[i])
    Tumor_class=main(img_png_path)
    Tumor_class_list.append(Tumor_class)

index_list=[]    
for index,string in enumerate(Tumor_class_list):
    if 'TUM' in string:
        index_list.append(index)
        
##获取final_tile_index
mucosa_tile_index=list(np.load(os.path.join('d:\\TCGA_mucosa_tile_index',"{}.npy".format(str(slide_num).zfill(4)))))
final_tile_index=[]
for seq in index_list:
  cancer_tile_index = mucosa_tile_index[seq]
  final_tile_index.append(cancer_tile_index)
  
tile_index_filename = os.path.join('d:\\TCGA_final_tile_index',"{}.npy".format(str(slide_num).zfill(4)))
np.save(tile_index_filename, final_tile_index)  ##保存WSI相对应肿瘤组织的坐标
#tile_index_num=list(np.load(os.path.join('d:\\TCGA_final_tile_index',"{}.npy".format(str(slide_num).zfill(4))))) ###读取坐标  


