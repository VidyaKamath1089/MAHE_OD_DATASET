import glob
import os  
import json 
import xml.etree.ElementTree as ET


unique_labels = {1: 'airconditioner', 2: 'air cooler', 3: 'airplane', 4: 'apple', 5: 'aquatic bird', 6: 'backpack', 7: 'bag', 8: 'ball', 9: 'banana', 10: 'bathtub', 
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

path = os.path.abspath("E:/MAHE-CUSTOM_DATASET_FOR_CONSTRAINED_MODEL_TRAINING/air conditioner")
annotation_path = "E:/MAHE-CUSTOM_DATASET_FOR_CONSTRAINED_MODEL_TRAINING/air conditioner"
dict ={}

for file in glob.glob(annotation_path + "*.xml"): 
  print(file)
  temp = file.split("/")[-1][:-3]
  if temp == 'classes.':continue
  image_name= temp + "jpg"
  image_path = path + "/" + image_name
  print(image_path)
  tree = ET.parse(file)
  root = tree.getroot()
  boxs =[] 
  labels =[]
  difficulties = []
  for object in root.iter('object'):
    label = object.find('name').text.lower().strip()
    bbox = object.find('bndbox')
    xmin = int(bbox.find('xmin').text) - 1
    ymin = int(bbox.find('ymin').text) - 1
    xmax = int(bbox.find('xmax').text) - 1
    ymax = int(bbox.find('ymax').text) - 1
    boxs.append([xmin, ymin, xmax, ymax])
    labels.append(unique_labels[ label  ])
    difficulties.append(0)
  dict[image_path] = {"boxes": boxs, "labels": labels, "difficulties": difficulties} 

   			


train_images = list()
train_objects = list()
test_images = list()
test_objects = list()

for key in dict:
	name = key.split("/")[-1]
	if name !=  'IMG_0504.jpg' and name != 'IMG_0505.jpg':   # we must splt our data and append
		train_images.append(key)
		train_objects.append(dict[key])
	else:
		test_images.append(key)
		test_objects.append(dict[key])




output_folder = './'
# Save to file
with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
    json.dump(train_images, j)


with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
    json.dump(train_objects, j)


with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
    json.dump(unique_labels, j)  # save label map too


with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
    json.dump(test_images, j)


with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
    json.dump(test_objects, j)
