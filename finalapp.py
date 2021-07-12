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
from image_processing import func
from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory
from werkzeug.utils import secure_filename

app=Flask(__name__)
#video
camera = cv2.VideoCapture(0)
#image
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'test1/dj'

def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413

@app.route('/image')
def index():
    print(1)
    files = os.listdir(app.config['UPLOAD_PATH'])
    
    return render_template('index.html', files=files)

    

    


@app.route('/image', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            return "Invalid image", 400
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
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
    model = pickle.load(open('f.sav','rb'))
    test1= pd.read_csv("test.csv",error_bad_lines=False)
    test1=test1.dropna()
    x1= np.array(test1)/255.
    pred12=model.predict(x1)
    #print(pred12[0])
    
    a=['1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
    for i in pred12:
        
        
        
      
        return render_template("index.html", prediction=a[i])
    return '', 204
#end of image

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
@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

@app.route('/')
def main1():
	return render_template('index21.html')           
@app.route('/video')
def index1():
    
    return render_template('index1.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(port=8082)