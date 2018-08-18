#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 13:32:15 2017

@author: JIL
"""
import cv2
import os
import numpy as np
from PIL import Image


print('This is the version for this')
print( cv2.__version__ )




#create a recognizer
projectdir = "/home/pi/Downloads/facedetection/"

recognizer = cv2.createLBPHFaceRecognizer()

#get the path of the samples
path = projectdir + 'dataSet'

#to get the corresponding ids
def getImagesWithId(path):
    #concatenate the root path with the image path
    #list all directories in the path and assigning it to f and appending it to the path with a slash
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    
    faces = []
    Ids = []
    
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8') #convert to gray again
        Id=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        print(Id)
        Ids.append(Id)
        cv2.imshow("training", faceNp)
        cv2.waitKey(10)
    return np.array(Ids), faces
    
Ids, faces = getImagesWithId(path)
recognizer.train(faces, Ids)
#this saves the training data to the recognizer folder
recognizer.save('recognizer/trainingDataNew.yml')
cv2.destroyAllWindows()

#print the recognizers
