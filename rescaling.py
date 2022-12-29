# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:06:07 2022

@author: MAHE
"""
#import random
#import xml.etree.ElementTree as ET
# import pandas as pd
# from PIL import Image
# from pathlib import Path
# import imagesize
# import numpy as np
import cv2
from matplotlib import pyplot as plt
import albumentations as A
#________________________________________________________________________________________________
# list of all category ids 
category_ids_all = list(range(1,181))
print(category_ids_all)

# We will use the mapping from category_id to the class name
# to visualize the class label for the bounding box on the image

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
print(category_id_to_name)

#Read images and bounding boxes in pascalvoc format

''' using single image here now. Need to code a loop for this once datasets are prepared  '''

path="D:/MOD016_172.jpg"
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


'''hardcoding now for testing single image, need to use element tree and pick values from xml file in future '''
bboxes = [[39,31,96,190] ,[114, 41, 166 ,189]]
category_ids = [16, 16]

 
#set the color for the bounding box and the label text 
BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


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
    
    
#visualize(image, bboxes, category_ids, category_id_to_name)
# to display original image 

# perform transformations here using Compose from albumentations.. 
transform = A.Compose([
    #A.Resize(width=450, height=450),
    #A.HorizontalFlip(p=1.0),
    #A.RandomBrightnessContrast(p=0.2),
    #A.CenterCrop(250,250)
    #A.Blur(blur_limit=7, always_apply=False, p=0.9)
    #A.GaussianBlur(blur_limit=(3,7), sigma_limit=0, always_apply=False, p=0.5)
    #A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, always_apply=False, mode='fast', p=0.5)
    #A.MotionBlur(blur_limit=8 , always_apply=True)
    #A.GaussNoise (var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5)
    #A.ISONoise(color_shift=(0.01, 0.05), intensity= (0.1, 0.5), always_apply=False, p=0.5)
    A.RandomRain (slant_lower=-10, slant_upper=10, drop_length=5, drop_width=1, drop_color=(0,0,0), blur_value=7, brightness_coefficient=0.7, rain_type= "drizzle", always_apply=True, p=1) # drizzle, heavy, torrential (raintypes)
    ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1,  label_fields=['category_ids']))

transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)

#to display the transformed image 
visualize(
    transformed['image'],
    transformed['bboxes'],
    transformed['category_ids'],
    category_id_to_name,
)