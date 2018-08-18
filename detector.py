#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 13:32:15 2017

@author: JIL
"""
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import sys
import cv2
import numpy as np
print('This is the version for this')
print( cv2.__version__ )

# Setup the camera
camera = PiCamera()
camera.resolution = ( 320, 240 )
camera.framerate = 40
rawCapture = PiRGBArray( camera, size=( 320, 240 ) )

#to call the classifier
faceDetect=cv2.CascadeClassifier('/home/pi/Downloads/facedetection/haarcascade_frontalface_default.xml') #works with grayscale images

#create recognizers
rec = cv2.createLBPHFaceRecognizer()

#load the training data
rec.load("/home/pi/Downloads/facedetection/recognizer/trainingDataNew.yml")

#initialize id to predict
Id = 0
#font = cv2.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL, 1,1,0,1)
font = cv2.FONT_HERSHEY_SIMPLEX

for frame in camera.capture_continuous( rawCapture, format="bgr", use_video_port=True ):
    img = frame.array
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #now that's gray, we can detect faces
    
    faces=faceDetect.detectMultiScale(gray,1.3,5); #detect faces and return coordinates of the face
    
    #to draw rectangles on the faces
    for(x,y,w,h) in faces: 
       cv2.rectangle(img,(x,y), (x+w, y+h), (0,255,0),2) #image, initial coordinates, their span, rectangle color, thickness of the rectangle
       
       #predict id of the face
       Id, conf = rec.predict(gray[y:y+h, x:x+w])
       print(Id)
       if(conf<50):
           if(Id==1):
               Id="JIl"
           elif(Id==2):
               Id="Sam"
           elif(Id==6):
               Id="Uche"
           elif(Id==5):
               Id="Tayo"
           else:
               Id="Unknown"
        
            
            
       #put text on the screen
    # working
       # cv2.putText(img, str(Id), (x,y+h), font, 2, 255, 2, cv2.FONT_HERSHEY_SIMPLEX)
    #/ working
       cv2.rectangle(img,(x-22,y-90), (x+w+22, y-22), (0,255,0),-1)
       cv2.putText(img, str(Id), (x,y-40), font, 2, (255,255,255), 3) 
       #img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA
       
       
    #showing the image
    cv2.imshow("Faces found", img);
    
    #a wait command
    if(cv2.waitKey(1) == ord('q')):
        break;
    
    #if(Id != 0):
    #    print('You have been recognized'+ str(Id))
    #    break;
     # Clear the stream in preparation for the next frame
    rawCapture.truncate( 0 )
