import cv2 as cv
import numpy as np


fire_cascade=cv.CascadeClassifier("fire_detection.xml")
video=cv.VideoCapture(0)

while True:
    ret,frame= video.read()
    gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    fire=fire_cascade.detectMultiScale(frame,1.2,5)
    for (x,y,w,h) in fire:
        cv.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)      
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w] 
        print("fire is detected")


    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break