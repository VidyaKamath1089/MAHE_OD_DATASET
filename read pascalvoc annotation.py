# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:48:28 2022

@author: MAHE
"""

import xml.etree.ElementTree as ET

path="D:/New folder/n00007846_6247.xml"
# parse xml file
tree = ET.parse(path) 
root = tree.getroot() # get root object

height = int(root.find("size")[0].text)
width = int(root.find("size")[1].text)
#channels = int(root.find("size")[2].text)


bbox_coordinates = []
for member in root.findall('object'):
    class_name = member[0].text # class name
        
    # bbox coordinates
    xmin = int(member[1][0].text)
    ymin = int(member[1][1].text)
    xmax = int(member[1][2].text)
    ymax = int(member[1][3].text)
    # store data in list
    bbox_coordinates.append([class_name, xmin, ymin, xmax, ymax])

print(bbox_coordinates)