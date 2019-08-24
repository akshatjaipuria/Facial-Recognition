import numpy as np
import cv2
from PIL import Image
import os

path='image_dataset'

recognizer=cv2.face.LBPHFaceRecognizer_create()
detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def get_image_and_labels(path):
    image_paths=[os.path.join(path,each) for each in os.listdir(path)]
    face_samples=[]
    ids=[]

    for image_path in image_paths:
        img=Image.open(image_path).convert('L')
        img_numpy=np.array(img)
        id = int(os.path.split(image_path)[-1].split(".")[1])
        face=detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in face:
            face_samples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return face_samples,ids

print("Training...")

faces,ids=get_image_and_labels(path)
recognizer.train(faces, np.array(ids))

recognizer.write('models/model_1.yml')

print("\n {0} faces trained. Exiting Program".format(len(np.unique(ids))))