import cv2
import numpy as np
from PIL import Image
import os

path = "data"

recognize = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    samplefaces = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert("L")
        img_numpy = np.array(PIL_img, "uint8")

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (w, x, y, z) in faces:
            samplefaces.append(img_numpy[x: x + z, w: w + y])
            ids.append(id)
    
    return samplefaces,ids

print("\n [INFO] Training Faces .....")
faces, ids = getImagesLabels(path)

recognize.train(faces, np.array(ids))

recognize.write('train/trainer.yml')

print("\n [INFO] {0} faces trained.".format(len(np.unique(ids))))