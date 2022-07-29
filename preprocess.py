# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:13:23 2022
@author: MAHE-Vidya Kamath
Code Assembly of all the Preprocessing Stages into a Single Program- preprocess.py 
"""

# Import Packages
import pandas as pd
import matplotlib.pyplot  as plt
from PIL import Image
from pathlib import Path
import imagesize
import numpy as np
import xml.etree.ElementTree as ET
import albumentations as A
import cv2
import random
from numba import vectorize, float32

category_ids_all = list(range(1,181)) # 180 object classes in MOD-2022
#print(category_ids_all)

""" We will use the mapping from category_id to the class name
 to visualize the class label for the bounding box on the image"""

category_id_to_name= {1: 'airconditioner', 2: 'air cooler', 3: 'airplane', 4: 'apple', 5: 'aquatic bird', 6: 'backpack', 7: 'bag', 8: 'ball', 9: 'banana', 10: 'bathtub', 
                       11: 'beans', 12: 'bear', 13: 'bed', 14: 'bench', 15: 'bicycle', 16: 'bird', 17: 'blender', 18: 'boat', 19: 'books', 20: 'bottle',
                       21: 'bowl', 22: 'broccoli', 23: 'broom', 24: 'bucket', 25: 'buffalo', 26: 'burger', 27: 'bus', 28: 'butterfly', 29: 'cake', 30: 'calculator',
                       31: 'camel',32: 'camera', 33: 'capsicum', 34: 'car', 35: 'carrot', 36: 'cat', 37: 'cauliflower', 38: 'cellphone', 39: 'chair', 40: 'chandelier', 
                       41: 'clock', 42: 'cockroach', 43: 'comb', 44: 'couch', 45: 'cow', 46: 'crab', 47: 'crocodile', 48: 'cup', 49: 'deer', 50: 'desk', 
                       51: 'dining table', 52: 'dog' , 53: 'dolphin', 54: 'donut', 55: 'door', 56: 'drangfly', 57: 'elephant', 58: 'eye glasses', 59: 'fan', 60: 'ferry', 
                       61: 'fire hydrant', 62: 'fish', 63: 'flowers', 64:'fork', 65: 'fried egg', 66: 'fries', 67: 'frisbee', 68: 'frog', 69: 'frying pan', 70: 'giraffe',
                       71: 'goat', 72: 'gorilla', 73: 'grapes', 74: 'hair dryer', 75: 'hammer', 76: 'hat', 77: 'head phones', 78: 'helicopter', 79: 'helmet', 80: 'horse', 
                       81: 'hot dog', 82: 'ice cream', 83: 'ipod', 84: 'kangaroo', 85: 'kayak', 86: 'keyboard', 87: 'keys', 88:'kite', 89: 'knife', 90: 'ladder', 
                       91: 'lamp', 92: 'laptop', 93: 'lemon', 94: 'lion', 95: 'lobster', 96: 'loofa', 97: 'mango', 98: 'mask', 99: 'mattress', 100: 'mice',
                       101: 'microwave', 102: 'mirror', 103: 'monkey', 104: 'motorbike', 105: 'mouse', 106: 'mushroom', 107: 'onion', 108: 'orange', 109: 'pan', 110: 'parking meter', 
                       111: 'pen', 112: 'penguin', 113: 'Person', 114: 'pie', 115: 'pig', 116: 'pillow', 117: 'pills', 118: 'pineapple', 119: 'pizza', 120: 'plate', 
                       121: 'pomegranate', 122: 'potato', 123: 'potted plant', 124: 'pumpkin', 125: 'rainbow', 126: 'refrigerator', 127: 'remote', 128: 'revolver', 129: 'rooster', 130: 'sandwich', 
                       131: 'school bus', 132: 'schooner', 133: 'scissors', 134: 'scooter', 135: 'screwdriver', 136: 'sheep', 137: 'shoes', 138: 'sink', 139: 'skateboard', 140: 'skyscraper', 
                       141: 'snake', 142: 'snowboard', 143: 'soap', 144: 'socks', 145: 'spaghetti', 146: 'spider', 147: 'spoon', 148: 'sportsball', 149: 'stapler', 150: 'steering wheel', 
                       151: 'stove', 152: 'strawberry', 153: 'street sign', 154: 'suitcase', 155: 'sushi', 156: 'table', 157: 'teddy bear', 158: 'tie', 159: 'tiger', 160: 'toaster', 
                       161: 'toilet seat', 162: 'tomato', 163: 'tooth brush', 164: 'traffic light', 165: 'train', 166: 'tree', 167: 'truck', 168: 'tv_monitor', 169: 'umbrella', 170: 'van', 
                       171: 'vase', 172: 'video projector', 173: 'wardrobe', 174: 'watch', 175: 'waterfall', 176: 'wheelchair', 177: 'windmill', 178: 'window', 179: 'xerox machine', 180: 'zebra'}
#print(category_id_to_name)

def get_key(val):
    for key,value in category_id_to_name.items():
        if val == value:
            return key
        
    return "not found"

#print(get_key('bird'))

#set the color for the bounding box and the label text 
BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

#Begin Function Definitions 
#___________________________________________________________________________________________________________________##
""" Define functions to Perform the Preprocessing Tasks """
__all__ = [
    "ReadImage", 
    "ReadBBox",
    "visualize", 
    "visualize_bbox",
    "preprocess",
    "AddNoise",
    "AddBlurs",
    "AddContrast",
    "ResizeImage",
    "HFlip",
    "CCrop",
     ]

def ReadImage(path):
    """ Reads the image specified in path and returns the image"""
    image = cv2.imread(path) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def ReadBBox(path):
    """
    Parameters
    ----------
    path : path to the xml file (in pascalvoc annotations format)
            Ex: path="D:/New folder/n00007846_6247.xml"

    Returns
    -------
    bboxes: the bounding box coordinates in the form [ xmin, ymin, xmax, ymax]
    category_ids: the corresponding ids of the objects present in the bounding boxes (1 to 180)
    height: height of the image
    width: width of the image
    channels: number of channels in the image
    """
    # parse xml file
    tree = ET.parse(path) 
    root = tree.getroot() # get root object

    height = int(root.find("size")[0].text) 
    width = int(root.find("size")[1].text)
    channels = int(root.find("size")[2].text)

    bboxes = []
    category_ids=[]
    for member in root.findall('object'):
        class_name = member[0].text # class name
            
        # bbox coordinates
        xmin = int(member[4][0].text)
        ymin = int(member[4][1].text)
        xmax = int(member[4][2].text)
        ymax = int(member[4][3].text)
        # store data in list
        bboxes.append([xmin, ymin, xmax, ymax])
        category_ids.append(get_key(class_name))
    return bboxes, category_ids, height, width, channels

        
def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    # x_min, y_min, w, h = bbox
    # x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)  # this is for coco 
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)       # modified to match pascal_voc
    
    
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img

def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    
def preprocess(
        image, 
        bboxes, 
        category_ids, 
        resize= False, 
        rheight= 256, 
        rwidth= 256, 
        flip= False, 
        contrast= False, 
        crop= False,
        blur_type= None, 
        blur_limit= 7,
        max_delta=4,
        noise_type= None, 
        var_limit=(10.0, 50.0), 
        intensity=(0.1, 0.5), 
        rain_type=None, 
        drop_color=(200,200,200),
        p=0.5,
        min_visibility= None, 
        min_area= None,
        ):
    """ 
     Parameters
    ----------
        image : image to be processed 
        bboxes : list of bounding boxes in the image specified by path - PASCAL VOC FORMAT USED 
        category_ids: corresponding category id of the Object class 
                ex: bboxes = [[39,31,96,190] ,[114, 41, 166 ,189]] 
                category_ids = [16, 16]
        resize: True- resize to rheight and rwidth
                False- no resizing
        rheight: resize height
        rwidth: resize width 
        flip:   True - horizontal flip 
                False- no flip 
        contrast: 
                True - apply contrast change
                False - no contrast changes
        crop: 
                True - crop the image to rheight, rwidth
                False - no cropping
        blur_types: type of blur to be added 
                    1. Normalblur
                    2. GaussianBlur
                    3. GlassBlur
                    4. MotionBlur 
        blur_type : 1, 2, 3 or 4
        blur_limit: 7
        max_delta : for glass blur
        noise_type: type of noise to be added 
                    1. GaussNoise
                    2. ISONoise
                    3. RandomRain
        var_limit: ( lower limit, upper limit) for gaussian noise
        intensity: (low,high) for ISO Noise
        rain_type: None, drizzle, heavy, torrential for random rain
        drop_color: (R, G, B) for random rain 
        p : probability of applying the effects 
    Returns
    ------- 
     transformed image 
        transformed['image']- holds the image
        transformed['bboxes'] - holds the recalculated Bounding boxes
        transformed['category_ids'] - holds the corresponding Category Ids of the BBoxes
    
    """
    transform=A.Compose([], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=min_visibility,  label_fields=['category_ids']))
    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
    
    if resize==True and crop==False:
        transformed= ResizeImage(transformed['image'], transformed['bboxes'], transformed['category_ids'], min_visibility, rheight, rwidth)
    elif resize==False and crop== True:
        transformed= CCrop(transformed['image'], transformed['bboxes'], transformed['category_ids'], min_visibility, rheight, rwidth)
    else: 
        raise ValueError(" Crop and resize both set to True. Choose One")
        
    if flip==True:
        transformed= HFlip(transformed['image'], transformed['bboxes'], transformed['category_ids'], min_visibility, p)
    
    if contrast==True:
        transformed= AddContrast(transformed['image'], transformed['bboxes'], transformed['category_ids'], min_visibility, p)
    
    transformed= AddNoise(transformed['image'], transformed['bboxes'], transformed['category_ids'], noise_type, min_visibility, var_limit, intensity, rain_type, drop_color, p)
    transformed= AddBlurs(transformed['image'], transformed['bboxes'], transformed['category_ids'], blur_type, min_visibility, blur_limit, max_delta, p)
        
    #to display the transformed image 
    visualize(
        transformed['image'],
        transformed['bboxes'],
        transformed['category_ids'],
        category_id_to_name,
    )
    
    return transformed['image'], transformed['bboxes'], transformed['category_ids']


def AddNoise(image, bboxes, category_ids, noise_type, min_visibility, var_limit, intensity, rain_type, drop_color, p ):
    """
    Applies the specified noise type in noise_type 

    Parameters
    ----------
    image : image to be processed 
    bboxes : list of bounding boxes in the image specified by path - PASCAL VOC FORMAT USED 
    category_ids: corresponding category id of the Object class
    noise_type : 1, 2 or 3
    min_visiblity - ex : 0.1 
    var_limit: ( lower limit, upper limit) for gaussian noise
    intensity: (low,high) for ISO Noise
    rain_type: None, drizzle, heavy, torrential for random rain
    drop_color: (R, G, B) for random rain 
    p : probability of applying the noise effect 
    
    Returns
    -------
    transformed image 
       transformed['image']- holds the image
       transformed['bboxes'] - holds the recalculated Bounding boxes
       transformed['category_ids'] - holds the corresponding Category Ids of the BBoxes

    """
    if noise_type==1:
        transform = A.Compose([
            A.GaussNoise (var_limit=var_limit, mean=0, per_channel=True, always_apply=False, p=p)
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=min_visibility,  label_fields=['category_ids']))
    elif noise_type==2:
        transform = A.Compose([
            A.ISONoise(color_shift=(0.01, 0.05), intensity= intensity, always_apply=False, p=p)
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1,  label_fields=['category_ids']))
    elif noise_type==3:
        transform = A.Compose([
            A.RandomRain (slant_lower=-10, slant_upper=10, drop_length=5, drop_width=1, drop_color=drop_color, blur_value=5, brightness_coefficient=0.7, rain_type= rain_type, always_apply=False, p=p) 
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1,  label_fields=['category_ids']))
    else:
        raise ValueError("Invalid Noise Type! Choose  1- GaussNoise   2- ISONoise  3- RandomRain")
        
    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
    return transformed

def AddBlurs(image, bboxes, category_ids, blur_type, min_visibility, blur_limit, max_delta, p):
    """
    Applies the specified blur type in blur_type 

    Parameters
    ----------
    image : image to be processed 
    bboxes : list of bounding boxes in the image specified by path - PASCAL VOC FORMAT USED 
    category_ids: corresponding category id of the Object class
    blur_type : 1, 2, 3 or 4
    min_visiblity - ex : 0.1 
    max_delta : for glass blur
    p : probability of applying the noise effect 
    
    Returns
    -------
    transformed image 
       transformed['image']- holds the image
       transformed['bboxes'] - holds the recalculated Bounding boxes
       transformed['category_ids'] - holds the corresponding Category Ids of the BBoxes

    """
    if blur_type==1:
        transform = A.Compose([
            A.Blur(blur_limit=blur_limit, always_apply=False, p=p)
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=min_visibility,  label_fields=['category_ids']))
    elif blur_type==2:
        transform = A.Compose([
            A.GaussianBlur(blur_limit=(3,blur_limit), sigma_limit=0, always_apply=False, p=p)
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1,  label_fields=['category_ids']))
    elif blur_type==3:
        transform = A.Compose([
            A.GlassBlur(sigma=0.7, max_delta=max_delta, iterations=2, always_apply=False, mode='fast', p=p)
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1,  label_fields=['category_ids']))
    elif blur_type==4:
         transform = A.Compose([
             A.MotionBlur(blur_limit=blur_limit , always_apply=False, p=p)
             ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1,  label_fields=['category_ids']))
    else:
        raise ValueError("Invalid Blur Type! Choose 1. Normalblur  2. GaussianBlur  3. GlassBlur  4. MotionBlur ")
        
    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
    return transformed
    
def AddContrast(image, bboxes, category_ids, min_visibility, p):
    transform = A.Compose([
        A.RandomBrightnessContrast(p=p),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=min_visibility,  label_fields=['category_ids']))
    
    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
    return transformed
    
def ResizeImage(image, bboxes, category_ids, min_visibility, rheight, rwidth):
    transform = A.Compose([
        A.Resize(width=rwidth, height=rheight),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=min_visibility,  label_fields=['category_ids']))
    
    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
    return transformed

def HFlip(image, bboxes, category_ids, min_visibility, p):
    transform = A.Compose([
        A.HorizontalFlip(p=p),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=min_visibility,  label_fields=['category_ids']))
    
    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
    return transformed

def CCrop(image, bboxes, category_ids, min_visibility, rheight, rwidth):
    transform = A.Compose([
        A.CenterCrop(rwidth, rheight)
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=min_visibility,  label_fields=['category_ids']))
    
    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
    return transformed
#______________________________________________________________________________________________________________________##
#Function Definitions End Here 
# Code Begins 

image= ReadImage(path= "C:/PhD/Objective 1/MOD016_99.jpg")
( bboxes, category_ids, height, width, channels) = ReadBBox(path="C:/PhD/Objective 1/MOD016_99.xml")
# visualize(image, bboxes, category_ids, category_id_to_name) # to visualize the original image 
(image, bboxes, category_ids) = preprocess(image, 
                                            bboxes, 
                                            category_ids,
                                            resize=True,
                                            rheight=124,
                                            rwidth=124,
                                            noise_type=3, 
                                            min_visibility=0.1,  
                                            blur_type=1, 
                                            blur_limit=4,  
                                            p=0.8 )

cv2. imwrite("C:/PhD/Objective 1/test.jpg",image) 