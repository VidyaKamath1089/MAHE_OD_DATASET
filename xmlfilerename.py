# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:52:56 2022

@author: MAHE
"""

import xml.etree.ElementTree as ET
path= "D:/New folder/n00007846_6247.xml"
tree = ET.parse(path) 
root = tree.getroot() # get root object
ele= root.find("filename")
string = ele.text

new=string.replace("jpg","JPEG")
print(new)

ele.text=new

tree.write(path)