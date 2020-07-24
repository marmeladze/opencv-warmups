import cv2
import os
import numpy as np
import face_recognition as fr
import random

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("train/train.yml")

names ={0:"George Clooney", 1:"Al Pacino", 2:"Brad Pitt", 3:"Matt Damon"}

test_path = 'test/images'
test_image = os.path.join(test_path, 'trio.jpg')
print(test_image)
# for image in test_images:
#     test_image = os.path.join(test_path, image)

test_image=cv2.imread(test_image)
faces_detected, gray_img = fr.face_recognition(test_image)

for faces in faces_detected:
    (x,y,w,h) = faces
    roi_gray=gray_img[y:y+h, x:x+h]
    _id, confidence=face_recognizer.predict(roi_gray)
    print("confidence: ", confidence)
    print("label: ", names[_id])
    fr.rectangle(test_image, faces)
    predicted_name=names[_id]
    if (confidence>100):
        continue
    fr.text(test_image, predicted_name, x, y)

resized_image = cv2.resize(test_image,(700,600))
cv2.imshow("Face Detection: ", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()