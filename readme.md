# Setup

```
virtualenv venv
git clone https://github.com/marmeladze/opencv-warmups.git
cd src && pip install -r requirements.txt
```

# Train

```
python train.py
```

Cat face detection 

```
python cat_face_detection.py
```
![screenshot](data/cat-detect.jpg)


Human face detection from live video 

```
python live_face_detection.py
```
![screenshot](data/live-face.png)


Face and eye detection from live video 

```
python face_and_eye_detection.py
```
![screenshot](data/eye-and-face-detect.png)


Face recognition & tagging on videos

```
python test_face_recog_on_video.py
# Produces undefined behaviour with qt5 
```


