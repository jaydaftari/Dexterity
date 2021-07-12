# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 23:06:51 2021

@author: dibya
"""

import numpy as np
import cv2
import os
import csv
from image_processing import func


path="test1"
a=[]
#training
for i in range(9216):
    a.append("pixel"+str(i))
    

#outputLine = a.tolist()

with open('test.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = a)
    writer.writeheader()

with open('test.csv', 'a') as csvfile:
    spamwriter = csv.writer(csvfile)
    
    
    for (dirpath,dirnames,filenames) in os.walk(path):
        for dirname in dirnames:
            print(dirname)
            for(direcpath,direcnames,files) in os.walk(path+"\\"+dirname):
                
                i=0
                for file in files:
                    actual_path=path+"\\\\"+dirname+"\\\\"+file
                    print(actual_path)
                    bw_image=func(actual_path)
                    flattened_sign_image=bw_image.flatten()
                    outputLine =np.array(flattened_sign_image).tolist()
                    
                    spamwriter.writerow(outputLine)
                    
                    i=i+1
                    
            
