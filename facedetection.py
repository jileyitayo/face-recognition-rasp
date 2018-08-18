#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 13:32:15 2017

"""
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import sys
import cv2
import numpy as np
print('This is the version for this')
print(cv2.__version__)

# Setup the camera
camera = PiCamera()
camera.resolution = ( 320, 240 )
camera.framerate = 60
rawCapture = PiRGBArray( camera, size=( 320, 240 ) )

# to call the classifier

faceDetect = cv2.CascadeClassifier('/home/pi/opencv/data/lbpcascades/lbpcascade_frontalface.xml')  # works with grayscale images

t_start = time.time()
fps = 0

### Main ######################################################################

for frame in camera.capture_continuous( rawCapture, format="bgr", use_video_port=True ):
    img = frame.array
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # now that's gray, we can detect faces

    # Look for faces and phones in the image using the loaded cascade file
    faces = faceDetect.detectMultiScale(gray)  # detect faces and return coordinates of the face

    print "Found " + str( len( faces ) ) + " face(s)"
     
    # to draw rectangles on the faces
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (100, 255, 100), 2)  # image, initial coordinates, their span, rectangle color, thickness of the rectangle
        cv2.putText( img, "Face No." + str( len( faces ) ), ( x, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 255 ), 2 )
   

    # Calculate and show the FPS
    fps = fps + 1
    sfps = fps / ( time.time() - t_start )
    cv2.putText( img, "FPS : " + str( int( sfps ) ), ( 10, 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 255 ), 2 )
    
    # showing the image
    cv2.imshow("Faces found", img)
    
    # a wait command
    if(cv2.waitKey(1) == ord('q')):
        break
    
    # Clear the stream in preparation for the next frame
    rawCapture.truncate( 0 )
