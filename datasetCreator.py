from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
print('This is the version for this')
print( cv2.__version__ )


#create dataset

projectdir = "/home/pi/Downloads/facedetection/"

# Setup the camera
camera = PiCamera()
camera.resolution = ( 320, 240 )
camera.framerate = 60
rawCapture = PiRGBArray( camera, size=( 320, 240 ) )

#to call the classifier
faceDetect = cv2.CascadeClassifier(projectdir + 'lbpcascade_frontalface.xml')  # works with grayscale images
cv2.namedWindow("Register User")

id=input('enter user id') #storing an identifier
sampleNum=0;


t_start = time.time()
fps = 0

### Main ######################################################################

for frame in camera.capture_continuous( rawCapture, format="bgr", use_video_port=True ):
    img = frame.array
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #now that's gray, we can detect faces
    
    faces=faceDetect.detectMultiScale(gray,1.3,5); #detect faces and return coordinates of the face
    
    #to draw rectangles on the faces
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1
        cv2.imwrite(projectdir + "dataSet/User."+str(id)+"."+str(sampleNum)+".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img,(x,y), (x+w, y+h), (0,255,0),2) #image, initial coordinates, their span, rectangle color, thickness of the rectangle
        #wait for 0.1s before the next loop
        cv2.waitKey(100)
        
        
    # Calculate and show the FPS
    fps = fps + 1
    sfps = fps / ( time.time() - t_start )
    cv2.putText( img, "FPS : " + str( int( sfps ) ), ( 10, 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 255 ), 2 )
    
    #showing the image
    cv2.imshow("Faces found", img);
    
    #a wait command
    cv2.waitKey(1)
    if(sampleNum>30):
        break;
    
# Clear the stream in preparation for the next frame
    rawCapture.truncate( 0 )