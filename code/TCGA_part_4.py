# -*- coding: utf-8 -*-
"""
Created on Sun May 19 19:21:48 2024

@author: Xin
"""
#TCGA get mean prob
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



##step three 获取每张WSI的平均prob
N_dict = {'N':0,'P':1}
N_status_dataframe=pd.read_excel('d:\\data\\TCGA_N_p_n.xlsx')
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

N_status_dataframe=pd.read_excel('d:\\data\\TCGA_N_p_n.xlsx')
N_status_array=N_status_dataframe.values

def get_tiles_num(slide_num):    
  base_file='d:\\data_sets\\test_TCGA\\' 
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
  base_file='d:\\data_sets\\test_TCGA\\'
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

#def get_mean_prob_list(starting,ending):
  mean_prob_list=[]
  for slide_num in range(starting,ending):
    mean_prob=get_mean_prob(slide_num)
    mean_prob_list.append(mean_prob)
  return mean_prob_list

df = pd.read_excel('d:\\paper\\data\\TCGA_D2.xlsx') 
lst=list(df['ID'])
def get_mean_prob_list():
  mean_prob_list=[]
  for slide_num in lst:
    mean_prob=get_mean_prob(slide_num)
    mean_prob_list.append(mean_prob)
  return mean_prob_list

#mean_prob_list_dataframe=pd.DataFrame(mean_prob_list)
#mean_prob_list_dataframe.to_excel('d:\\TCGA_mean_prob_list_dataframe.xlsx',index=False)   



