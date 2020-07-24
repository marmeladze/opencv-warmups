import cv2
import os
import numpy as np
import face_recognition as fr

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("train/train.yml")
names ={0:"George Clooney", 1:"Al Pacino", 2:"Brad Pitt", 3:"Matt Damon"}


cap = cv2.VideoCapture("test/videos/ot-video-1.mp4")

while True:
    ret, test_image=cap.read()
    faces_detected, gray_image = fr.face_recognition(test_image)
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_image,(x,y),(x+w,y+h),(255,255,255), thickness=4)    
    resize_img = cv2.resize(test_image,(500,500))
    cv2.imshow("Face Detection: ", resize_img)
    cv2.waitKey(10)
    for faces in faces_detected:
        (x,y,w,h) = faces
        roi_gray = gray_image[y:y+w, x:x+h]
        _id, confidence = face_recognizer.predict(roi_gray)
        print("Confidence: ", confidence)
        print("Label: ", names[_id])
        fr.rectangle(test_image, faces)
        predicted_name = name[_id]
        if confidence < 150:
            fr.text(test_image,predicted_name,x,y)
    resize_img = cv2.resize(test_image,(500,500))
    cv2.imshow("Face recognition Tutorial: ", resize_img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
