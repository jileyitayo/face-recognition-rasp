#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 13:32:15 2017

@author: JIL
"""
import cv2
import numpy as np
print('This is the version for this')
print(cv2.__version__)


# create dataset
# run script called dataser creator

# train the recognizer
# create script called trainer

# detector
# create a script called  detector


# to call the classifier
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # works with grayscale images
cv2.namedWindow("preview")

print(cv2)

cam = cv2.VideoCapture(0)  # webcam is most times 0
rec = cv2.face.createLBPHFaceRecognizer()
# rec = cv2.face.createEigenFaceRecognizer()
rec.load('recognizer/trainingData.yml')


id = 0
# font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL, 5, 1, 0, 4)
font = cv2.FONT_HERSHEY_SIMPLEX

# rval, frame = cam.read()
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # now that's gray, we can detect faces

    faces = faceDetect.detectMultiScale(gray, 1.3, 5)  # detect faces and return coordinates of the face

    # to draw rectangles on the faces
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # image, initial coordinates, their span, rectangle color, thickness of the rectangle
        id = rec.predict(gray[y: y + h, x: x + w])
        cv2.putText(cv2.cv.fromarray(img), str(id), 4, (x, y + h), font, 255, 2)

    # showing the image
    cv2.imshow("Faces found", img)

    # a wait command
    if(cv2.waitKey(1) == ord('q')):
        break

# release the camera
cam.release()
cv2.destroyAllWindows()
