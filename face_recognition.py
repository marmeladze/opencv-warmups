import cv2
import os
import numpy as np
from PIL import Image

def face_recognition(test_image):
    gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier((os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"))
    faces = face_haar_cascade.detectMultiScale(gray_image, scaleFactor=1.40, minNeighbors=5)
    return faces, gray_image



def labels(directory):
    faces=[]
    faces_id=[]
    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("dot file")
                continue

            id=os.path.basename(path)
            image_path=os.path.join(path, filename)
            print("image_path:", image_path)
            print("id: ", id)
            test_image=cv2.imread(image_path)
            if test_image is None:
                print("IO error")
                continue
            face_rect, gray_image=face_recognition(test_image)
            if len(face_rect) != 1:
                continue

            (x,y,w,h)=face_rect[0]
 
            r_gray=gray_image[y:y+w, x:x+h]
            faces.append(r_gray)
            faces_id.append(int(id))
    return faces, faces_id

def training(faces,faces_id):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faces_id))
    face_recognizer.save("train/train.yml")
    return face_recognizer

def rectangle(test_image, face):
    (x,y,w,h) = face
    cv2.rectangle(test_image, (x,y), (x+w,y+h), (255,255,255),thickness=2)

def text(test_image, text, x, y):
    cv2.putText(test_image, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,1 ,(255,0,0), 1)