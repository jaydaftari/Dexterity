# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 21:56:19 2021

@author: dibya
"""

from flask import Flask, render_template, Response
import cv2
from p2 import func1
import pandas as pd
import pickle
import numpy as np
import imghdr
import csv
import os

app=Flask(__name__)
camera = cv2.VideoCapture(0)
#p="value"

def gen_frames():  
    while (1):
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            
           
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
                
                
                
                bw_image=func1(frame)
                flattened_sign_image=bw_image.flatten()
                outputLine =np.array(flattened_sign_image).tolist()
                
                spamwriter.writerow(outputLine)
                
                i=i+1
            model = pickle.load(open('f.sav','rb'))
            test1= pd.read_csv("test.csv",error_bad_lines=False)
            test1=test1.dropna()
            x1= np.array(test1)/255.
            pred12=model.predict(x1)
            #print(pred12[0])
            
            a=['1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
            for i in pred12:
                
                
                
              
                print(a[i])
                #return render_template("index.html", prediction=a[i])
                
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
@app.route('/')
def index():
    
    return render_template('index1.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(port=8082)