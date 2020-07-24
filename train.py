import face_recognition as fr

faces, ids = fr.labels('train/images/')
fr.training(faces, ids)
