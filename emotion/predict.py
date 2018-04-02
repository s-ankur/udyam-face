import cv2
import glob,os
import random
import numpy as np

class FaceNotFound(Exception):
    
    

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_detector_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
face_detector_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
face_detector_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

fishface = cv2.face.createFisherFaceRecognizer()  
fishface.load('tmp.yaml')

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions


def extract_face(img):
    frame=cv2.imread(img)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    face = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_two = face_detector_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_three = face_detector_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_four = face_detector_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    
     if len(face) == 1:
        facefeatures = face
    elif len(face_two) == 1:
        facefeatures = face_two
    elif len(face_three) == 1:
        facefeatures = face_three
    elif len(face_four) == 1:
        facefeatures = face_four
    else:
        raise FaceNotFound
    
    for (x, y, w, h) in facefeatures: 
        gray = gray[y:y+h, x:x+w] 
    gray = cv2.resize(gray, (350, 350)) 
    return gray

try:
    while True:
        face=extract_face(input("Input Image: "))
        predicted_class=fishface.predict(face)
        print(emotions[predicted_class])
except KeyboardInterrupt:
    print("Quiter")
