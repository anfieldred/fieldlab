# -*- coding: utf-8 -*-
"""
Created on Sun May 19 18:18:00 2024

@author: Xin
"""
# For geting TCGA mucosa tiles
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
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_dilation, disk
import datetime

# parameter值
folder="d:\\TCGA"
SCALE_FACTOR=32
img_tile_size=16   #16*32=512(tile size)     32*32=1024
tile_size=512
overlap=0
sample_size = 512    
starting=1
ending=248
class Time:
  """
  Class for displaying elapsed time.
  """

  def __init__(self):
    self.start = datetime.datetime.now()

  def elapsed_display(self):
    time_elapsed = self.elapsed()
    print("Time elapsed: " + str(time_elapsed))

  def elapsed(self):
    self.end = datetime.datetime.now()
    time_elapsed = self.end - self.start
    return time_elapsed


# 1.Open Whole-Slide Image
def open_slide(slide_num, folder):
  filename = os.path.join(folder, str(slide_num).zfill(4)+'.svs')                     
  try:
    slide = openslide.open_slide(filename)
  except OpenSlideError:
    slide = None
  except FileNotFoundError:
    slide = None
  return slide

# 2.Convert a WSI training slide to a scaled-down PIL image and save
def slide_to_scaled_pil_image(slide_num, folder):
  """
  Convert a WSI training slide to a scaled-down PIL image.

  Args:
    slide_number: The slide number.

  Returns:
    Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
  """
  slide = open_slide(slide_num, folder)

  large_w, large_h = slide.dimensions
  new_w = math.floor(large_w / SCALE_FACTOR)
  new_h = math.floor(large_h / SCALE_FACTOR)
  level = slide.get_best_level_for_downsample(SCALE_FACTOR)
  whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
  whole_slide_image = whole_slide_image.convert("RGB")
  img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
  return img 

##img.save(os.path.join('d:\\img\\', str(slide_num).zfill(4)+'.png') ,'png')

def open_img(slide_num):
   filename = os.path.join('d:\\TCGA_img', str(slide_num).zfill(4)+'.png')   
   img = Image.open(filename)
   return img


#3.标记img后生成img_mark并打开    
def open_img_mark(slide_num):
    filename = os.path.join('d:\\TCGA_img_mark', str(slide_num).zfill(4)+'.png')  
    img_mark = Image.open(filename)
    return img_mark

#4.Create a mask to filter out pixels(remaining black)
## Generate tile indices for an img_mark   img_tile_size = 16
def get_img_tile_indices(slide_num, img_tile_size):
  """
  Generate all possible tile indices for a img_mark.
  

  Returns:
    A list of (slide_num, tile_size, col, row)
    integer index tuples representing possible tiles to extract.
  """
  
  ### Generate all possible (col, row) tile index tuples.
  img_mark = open_img_mark(slide_num)
  cols = math.ceil(img_mark.size[0]/img_tile_size)
  rows = math.ceil(img_mark.size[1]/img_tile_size)
  img_tile_indices = [(col, row)
                  for col in range(cols) for row in range(rows)]
  return img_tile_indices


##cut img_mark      'numpy.ndarray' object has no attribute 'crop'将img_mark切割成16x16大小的图片
def cut_img_mark(slide_num, img_tile_size):
  img_mark = open_img_mark(slide_num)  
  width, height = img_mark.size
  cols = math.ceil(img_mark.size[0]/img_tile_size)
  rows = math.ceil(img_mark.size[1]/img_tile_size)
  box_list = []  
  item_width = img_tile_size
  ### (left, upper, right, lower) 
  for r in range(0,rows):###两重循环，生成图片基于原图的位置 
    for c in range(0,cols):      
      box = (c*item_width,r*item_width,(c+1)*item_width,(r+1)*item_width)
      box_list.append(box) 
  img_mark_tile_list = [img_mark.crop(box) for box in box_list]  
  return img_mark_tile_list

##cut img
def cut_img(slide_num, img_tile_size):
  img = open_img(slide_num)  
  width, height = img.size
  cols = math.ceil(img.size[0]/img_tile_size)
  rows = math.ceil(img.size[1]/img_tile_size)
  box_list = []  
  item_width = img_tile_size
  # (left, upper, right, lower) 
  for r in range(0,rows):#两重循环，生成图片基于原图的位置 
    for c in range(0,cols):      
      box = (c*item_width,r*item_width,(c+1)*item_width,(r+1)*item_width)
      box_list.append(box) 
  img_tile_list = [img.crop(box) for box in box_list]  
  return img_tile_list

##生成每一个img_mark_tile图片名称 img_mark_tile 在 img_mark_tile_list图片中
def get_name_list(slide_num, img_tile_size):
  img_mark = open_img_mark(slide_num)  
  cols = math.ceil(img_mark.size[0]/img_tile_size)
  rows = math.ceil(img_mark.size[1]/img_tile_size)
  name_list = []  
  item_width = img_tile_size  
  for r in range(0,rows): 
    for c in range(0,cols):      
      name = '{}_{}'.format(c,r)
      name_list.append(name)
  return name_list

##生成img_mark_tile图片坐标
def get_list_index(slide_num, img_tile_size):
    img_mark = open_img_mark(slide_num)
    cols = math.ceil(img_mark.size[0]/img_tile_size)
    rows = math.ceil(img_mark.size[1]/img_tile_size)
    list_index = []    
    for r in range(0,rows): 
      for c in range(0,cols):      
        index = (c,r)
        list_index.append(index)
    return list_index

##提取img_mark标记黑色的坐标及序列 黑色面积大于90% 即32*32*3*0.9=2764    OR   16*16*3*0.9=691
def keep_img_mark_tile_index(slide_num, seq):
    img_mark = open_img_mark(slide_num)
    img_n= img_mark_tile_list[seq]
    img_n_np=np.asarray(img_n)
    if np.count_nonzero(img_n_np<=20)>=691:    
        img_mark_tile_index=list_index[seq]
        return (seq, img_mark_tile_index)  
    else:
      return False
def get_img_mark_tile_index_result(slide_num):  
  img_mark_tile_index_result=[]
  for seq in range(len(list_index)):
    if keep_img_mark_tile_index(slide_num, seq):
        img_mark_tile_index_result.append(keep_img_mark_tile_index(slide_num, seq))
  return img_mark_tile_index_result

def get_img_mark_seq(slide_num):        
  img_mark_seq_index=np.array(img_mark_tile_index_result)
  img_mark_seq=list(img_mark_seq_index[:,0])   
  return img_mark_seq

def get_img_mark_index(slide_num):        
  img_mark_seq_index=np.array(img_mark_tile_index_result)  
  img_mark_index=list(img_mark_seq_index[:,1]) 
  return img_mark_index

##将img_mark标记黑色的坐标及序列代入img中生成相应坐标的img_tile img_tile在img_tile_list中
##坐标适用于generator格式,序列适用于img_list格式
def get_img_tile_np_list(slide_num):
  img_tile_np_list=[]
  for seq in get_img_mark_seq(slide_num):
    img_tile_n=img_tile_list[seq]
    img_tile_np=np.asarray(img_tile_n)
    img_tile_np_list.append(img_tile_np)
  return img_tile_np_list


#5.Determine if a img_tile should be kept 缩小32x 第2次过滤
def get_optical_density(tile):
  """
  Convert a tile to optical density values.

  Args:
    tile: A 3D NumPy array of shape (tile_size, tile_size, channels).

  Returns:
    A 3D NumPy array of shape (tile_size, tile_size, channels)
    representing optical density values.
  """
  tile = tile.astype(np.float64)
  #od = -np.log10(tile/255 + 1e-8)
  od = -np.log((tile+1)/240)
  return od


    
def keep_img_tile(slide_num, tile):
  if tile.shape[0:2] == (img_tile_size, img_tile_size):
    tile_orig = tile

    # Check 1
    # Convert 3D RGB image to 2D grayscale image, from
    # 0 (dense tissue) to 1 (plain background).
    tile = rgb2gray(tile)
    # 8-bit depth complement, from 1 (dense tissue)
    # to 0 (plain background).
    tile = 1 - tile
    # Canny edge detection with hysteresis thresholding.
    # This returns a binary map of edges, with 1 equal to
    # an edge. The idea is that tissue would be full of
    # edges, while background would not.
    tile = canny(tile)
    # Binary closing, which is a dilation followed by
    # an erosion. This removes small dark spots, which
    # helps remove noise in the background.
    tile = binary_closing(tile, disk(10))
    # Binary dilation, which enlarges bright areas,
    # and shrinks dark areas. This helps fill in holes
    # within regions of tissue.
    tile = binary_dilation(tile, disk(10))
    # Fill remaining holes within regions of tissue.
    tile = binary_fill_holes(tile)
    # Calculate percentage of tissue coverage.
    percentage = tile.mean()
    check1 = percentage >= 0.99

    # Check 2
    # Convert to optical density values
    tile = get_optical_density(tile_orig)
    # Threshold at beta
    beta = 0.15
    tile = np.min(tile, axis=2) >= beta
    # Apply morphology for same reasons as above.
    tile = binary_closing(tile, disk(2))
    tile = binary_dilation(tile, disk(2))
    tile = binary_fill_holes(tile)
    percentage = tile.mean()
    check2 = percentage >= 0.99

    return check1 and check2
  else:
    return False        

def keep_img_tile_index(slide_num, seq):
  tile = img_tile_np_list[seq]
  if keep_img_tile(slide_num, tile):
        final_tile=img_mark_index[seq]
        return final_tile  
  else:
    return False
##生成img_tile_index在缩小32x图片上完成2次过滤
def get_img_tile_index_result(slide_num):
  img_tile_index_result=[]
  for seq in range(len(img_mark_index)):
    if keep_img_tile_index(slide_num, seq):
        img_tile_index_result.append(keep_img_tile_index(slide_num, seq))
  return img_tile_index_result
        
##生成粘膜层tiles(512 x)
class Time:
  """
  Class for displaying elapsed time.
  """

  def __init__(self):
    self.start = datetime.datetime.now()

  def elapsed_display(self):
    time_elapsed = self.elapsed()
    print("Time elapsed: " + str(time_elapsed))

  def elapsed(self):
    self.end = datetime.datetime.now()
    time_elapsed = self.end - self.start
    return time_elapsed

# 1.Open Whole-Slide Image
def open_slide(slide_num, folder):
  filename = os.path.join(folder, str(slide_num).zfill(4)+'.svs')                     
  try:
    slide = openslide.open_slide(filename)
  except OpenSlideError:
    slide = None
  except FileNotFoundError:
    slide = None
  return slide
         
# 2.Create Tile Generator坐标返回到generator上
def create_tile_generator(slide_num, tile_size, overlap):
  """
  Create a tile generator for the given slide.

  This generator is able to extract tiles from the overall
  whole-slide image.

  Args:
    slide: An OpenSlide object representing a whole-slide image.
    tile_size: The width and height of a square tile to be generated.
    overlap: Number of pixels by which to overlap the tiles.

  Returns:
    A DeepZoomGenerator object representing the tile generator. Each
    extracted tile is a PIL Image with shape
    (tile_size, tile_size, channels).
    Note: This generator is not a true "Python generator function", but
    rather is an object that is capable of extracting individual tiles.
  """
  slide = open_slide(slide_num, folder)
  generator = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=True)
  return generator
##generator=create_tile_generator(slide, 1024, overlap=0)
##print(‘切分的每层块数’，generator.level_tiles)  print(‘生成的层数’，generator.level_count)  print(‘切分成的块数’，generator.tile_count) print(‘每层尺寸大小’，generator.level_dimensions)
##tile=generator.get_tile(18,(24,28))  plt.imshow(tile)

# Determine 40x Magnification Zoom Level

def get_40x_zoom_level(slide_num, tile_size, overlap):
  """
  Return the zoom level that corresponds to a 20x magnification.

  The generator can extract tiles from multiple zoom levels,
  downsampling by a factor of 2 per level from highest to lowest
  resolution.

  Args:
    slide: An OpenSlide object representing a whole-slide image.
    generator: A DeepZoomGenerator object representing a tile generator.
      Note: This generator is not a true "Python generator function",
      but rather is an object that is capable of extracting individual
      tiles.

  Returns:
    Zoom level corresponding to a 20x magnification, or as close as
    possible.
  """
  generator = create_tile_generator(slide_num, tile_size, overlap)
  highest_zoom_level = generator.level_count - 1  # 0-based indexing
  level = highest_zoom_level
  return level

##在给定slide_num的wsi上生成给定坐标后tiles图片
def cut_slide_num_tiles(slide_num):
    slide_num_tiles = [] 
    for tile_index in list(np.load(os.path.join('d:\\TCGA_mucosa_tile_index',"{}.npy".format(str(slide_num).zfill(4))) )):      
        tile = np.asarray(generator.get_tile(level, tile_index))
        slide_num_tiles.append(tile)  
    return slide_num_tiles
##得到slide_num相应坐标tile图片的名称
def get_slide_num_tile_names(slide_num):
  slide_num_tile_names = []  
  for seq in range(len(tile_index_list[slide_num-1])):      
    tile_name = '{}_{}'.format(list(np.load(os.path.join('d:\\TCGA_mucosa_tile_index',"{}.npy".format(str(slide_num).zfill(4))) ))[seq][0],list(np.load(os.path.join('d:\\TCGA_mucosa_tile_index',"{}.npy".format(str(slide_num).zfill(4))) ))[seq][1])
    slide_num_tile_names.append(tile_name)
  return slide_num_tile_names

##切割tile成samples   一个1024x tile可切成   4块512x samples       16块256x samples         
def process_tile(tile, sample_size, grayscale):  
  if grayscale:
    tile = rgb2gray(tile)[:, :, np.newaxis]  # Grayscale
    # Save disk space and future IO time by converting from [0,1] to [0,255],
    # at the expense of some minor loss of information.
    tile = np.round(tile * 255).astype("uint8")
  x, y, ch = tile.shape
  # 1. Reshape into a 5D array of (num_x, sample_size_x, num_y, sample_size_y, ch), where
  # num_x and num_y are the number of chopped tiles on the x and y axes, respectively.
  # 2. Swap sample_size_x and num_y axes to create
  # (num_x, num_y, sample_size_x, sample_size_y, ch).
  # 3. Combine num_x and num_y into single axis, returning
  # (num_samples, sample_size_x, sample_size_y, ch).
  samples = (tile.reshape((x // sample_size, sample_size, y // sample_size, sample_size, ch))
                 .swapaxes(1,2)
                 .reshape((-1, sample_size, sample_size, ch)))
  samples = [sample for sample in list(samples)]
  return samples
##得到1个slide_num tiles的samples（4个sample）
def get_slide_num_samples(slide_num):
    slide_num_samples=[]
    for tile in slide_num_tiles:
        samples = process_tile(tile, sample_size, False)
        slide_num_samples.append(samples)
    return slide_num_samples
##将一个slide_num的samples(4个sample)展开  
def get_slide_num_sample(slide_num):   
    slide_num_sample_list=[]
    for samples in get_slide_num_samples(slide_num):
      for sample in samples:
          slide_num_sample_list.append(sample) 
    return slide_num_sample_list

def normalize_staining(sample, beta=0.15, alpha=1, light_intensity=255):
  """
  Normalize the staining of H&E histology slides.

  This function normalizes the staining of H&E histology slides.

  References:
    - Macenko, Marc, et al. "A method for normalizing histology slides
    for quantitative analysis." Biomedical Imaging: From Nano to Macro,
    2009.  ISBI'09. IEEE International Symposium on. IEEE, 2009.
      - http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
    - https://github.com/mitkovetta/staining-normalization

  Args:
    sample_tuple: A (slide_num, sample) tuple, where slide_num is an
      integer, and sample is a 3D NumPy array of shape (H,W,C).

  Returns:
    A (slide_num, sample) tuple, where the sample is a 3D NumPy array
    of shape (H,W,C) that has been stain normalized.
  """
  # Setup.
  
  x = np.asarray(sample)
  h, w, c = x.shape
  x = x.reshape(-1, c).astype(np.float64)  # shape (H*W, C)

  # Reference stain vectors and stain saturations.  We will normalize all slides
  # to these references.  To create these, grab the stain vectors and stain
  # saturations from a desirable slide.

  # Values in reference implementation for use with eigendecomposition approach, natural log,
  # and `light_intensity=240`.
  #stain_ref = np.array([0.5626, 0.2159, 0.7201, 0.8012, 0.4062, 0.5581]).reshape(3,2)
  #max_sat_ref = np.array([1.9705, 1.0308]).reshape(2,1)

  # SVD w/ log10, and `light_intensity=255`.
  stain_ref = (np.array([0.54598845, 0.322116, 0.72385198, 0.76419107, 0.42182333, 0.55879629])
                 .reshape(3,2))
  max_sat_ref = np.array([0.82791151, 0.61137274]).reshape(2,1)

  # Convert RGB to OD.
  # Note: The original paper used log10, and the reference implementation used the natural log.
  #OD = -np.log((x+1)/light_intensity)  # shape (H*W, C)
  OD = -np.log10(x/light_intensity + 1e-8)

  # Remove data with OD intensity less than beta.
  # I.e. remove transparent pixels.
  # Note: This needs to be checked per channel, rather than
  # taking an average over all channels for a given pixel.
  OD_thresh = OD[np.all(OD >= beta, 1), :]  # shape (K, C)

  # Calculate eigenvectors.
  # Note: We can either use eigenvector decomposition, or SVD.
  #eigvals, eigvecs = np.linalg.eig(np.cov(OD_thresh.T))  # np.cov results in inf/nans
  U, s, V = np.linalg.svd(OD_thresh, full_matrices=False)

  # Extract two largest eigenvectors.
  # Note: We swap the sign of the eigvecs here to be consistent
  # with other implementations.  Both +/- eigvecs are valid, with
  # the same eigenvalue, so this is okay.
  #top_eigvecs = eigvecs[:, np.argsort(eigvals)[-2:]] * -1
  top_eigvecs = V[0:2, :].T * -1  # shape (C, 2)

  # Project thresholded optical density values onto plane spanned by
  # 2 largest eigenvectors.
  proj = np.dot(OD_thresh, top_eigvecs)  # shape (K, 2)

  # Calculate angle of each point wrt the first plane direction.
  # Note: the parameters are `np.arctan2(y, x)`
  angles = np.arctan2(proj[:, 1], proj[:, 0])  # shape (K,)

  # Find robust extremes (a and 100-a percentiles) of the angle.
  min_angle = np.percentile(angles, alpha)
  max_angle = np.percentile(angles, 100-alpha)

  # Convert min/max vectors (extremes) back to optimal stains in OD space.
  # This computes a set of axes for each angle onto which we can project
  # the top eigenvectors.  This assumes that the projected values have
  # been normalized to unit length.
  extreme_angles = np.array(
    [[np.cos(min_angle), np.cos(max_angle)],
     [np.sin(min_angle), np.sin(max_angle)]]
  )  # shape (2,2)
  stains = np.dot(top_eigvecs, extreme_angles)  # shape (C, 2)

  # Merge vectors with hematoxylin first, and eosin second, as a heuristic.
  if stains[0, 0] < stains[0, 1]:
    stains[:, [0, 1]] = stains[:, [1, 0]]  # swap columns

  # Calculate saturations of each stain.
  # Note: Here, we solve
  #    OD = VS
  #     S = V^{-1}OD
  # where `OD` is the matrix of optical density values of our image,
  # `V` is the matrix of stain vectors, and `S` is the matrix of stain
  # saturations.  Since this is an overdetermined system, we use the
  # least squares solver, rather than a direct solve.
  sats, _, _, _ = np.linalg.lstsq(stains, OD.T)

  # Normalize stain saturations to have same pseudo-maximum based on
  # a reference max saturation.
  max_sat = np.percentile(sats, 99, axis=1, keepdims=True)
  sats = sats / max_sat * max_sat_ref

  # Compute optimal OD values.
  OD_norm = np.dot(stain_ref, sats)

  # Recreate image.
  # Note: If the image is immediately converted to uint8 with `.astype(np.uint8)`, it will
  # not return the correct values due to the initital values being outside of [0,255].
  # To fix this, we round to the nearest integer, and then clip to [0,255], which is the
  # same behavior as Matlab.
  #x_norm = np.exp(OD_norm) * light_intensity  # natural log approach
  x_norm = 10**(-OD_norm) * light_intensity - 1e-8  # log10 approach
  x_norm = np.clip(np.round(x_norm), 0, 255).astype(np.uint8)
  x_norm = x_norm.astype(np.uint8)
  x_norm = x_norm.T.reshape(h,w,c)
  return x_norm

def get_x_norm_list(slide_num):
    x_norm_list=[]
    for  sample in slide_num_sample_list:
        x_norm = normalize_staining(sample, beta=0.15, alpha=1, light_intensity=255)
        x_norm_list.append(x_norm)   
    return x_norm_list

##保存x_norm图片            
def np_to_pil(np_img):
  """
  Convert a NumPy array to a PIL Image.

  Args:
    np_img: The image represented as a NumPy array.

  Returns:
     The NumPy array converted to a PIL Image.
  """
  if np_img.dtype == "bool":
    np_img = np_img.astype("uint8") * 255
  elif np_img.dtype == "float64":
    np_img = (np_img * 255).astype("uint8")
  return Image.fromarray(np_img)

#Example生成samples图片及验证数量是否正确
slide_num=1
img=open_img(slide_num)
img_mark=open_img_mark(slide_num)
img_tile_indices=get_img_tile_indices(slide_num, img_tile_size)
img_mark_tile_list=cut_img_mark(slide_num, img_tile_size)
img_tile_list=cut_img(slide_num, img_tile_size)
list_index=get_list_index(slide_num, img_tile_size)
img_mark_tile_index_result=get_img_mark_tile_index_result(slide_num)
img_mark_seq=get_img_mark_seq(slide_num)  
img_mark_index=get_img_mark_index(slide_num)  
img_tile_np_list=get_img_tile_np_list(slide_num)  
img_tile_index_result=get_img_tile_index_result(slide_num) 
tile_index_filename = os.path.join('d:\\TCGA_mucosa_tile_index',"{}.npy".format(str(slide_num).zfill(4)))
np.save(tile_index_filename, img_tile_index_result) 
   
slide=open_slide(slide_num,folder)
generator=create_tile_generator(slide_num, tile_size, overlap)
level=get_40x_zoom_level(slide_num, tile_size, overlap)
slide_num_tiles=cut_slide_num_tiles(slide_num)
slide_num_samples=get_slide_num_samples(slide_num)
slide_num_sample_list=get_slide_num_sample(slide_num)
x_norm_list=get_x_norm_list(slide_num)  
x_norm_name_list=list(range(len(x_norm_list)))  
x_norm_name_list.reverse()
out='d:\\TCGA_mucosa_tile_png\\'+str(slide_num).zfill(4)+'\\' ##如'd:\\mucosa_tile_png\\0001\\'
for x_norm in x_norm_list: 
     x_norm = np_to_pil(x_norm)    
     x_norm.save(out+str(x_norm_name_list.pop()).zfill(5) + '.png','png')  ##如'd:\\mucosa_tile_png\\0001\\1.png'

