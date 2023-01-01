# Importing libraries

import numpy as np
import cv2 as cv 
from PIL import Image
import time 

# Cascade Clissifier

face_casc = cv.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')  # it detects only frontal face
eye_casc = cv.CascadeClassifier('cascades/haarcascade_eye.xml')

# Face detection in a Video Capture

cap = cv.VideoCapture(0)   # 0 for laptop camera
if not cap.isOpened():
    print("Cannot open camera")
    exit()
       
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)   # This kind of cascade classifier works only on gray images
    faces= face_casc.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    
    for(x,y,w,h) in faces : 
        eye=gray[y:y+h, x:x+h]  
        eyes= eye_casc(eye, scaleFactor=1.5, minNeighbors=5)
        for(x1,y1,w1,h1) in eyes : 
            cv.rectangle(frame,(x1,y1),(x1+w1, y1+h1), (255,0,0))   # recadrer le visage
            cv.imshow('frame', frame)
            
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()