import cv2
import os
import numpy as np
import face_recognition as fcv

test_image = cv2.imread("test/images/trio.jpg")

face_detect, gray_image = fcv.face_recognition(test_image)

print("face Detected : ", face_detect)

for (x,y,w,h) in face_detect:
    cv2.rectangle(test_image,(x,y),(x+w,y+h),(255,255,255), thickness=2)

resize = cv2.resize(test_image,(500,500))

cv2.imshow("Face Detection :", resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
