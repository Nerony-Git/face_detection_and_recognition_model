import cv2
import numpy as np
import os


recognize = cv2.face.LBPHFaceRecognizer_create()
recognize.read('train/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
cascadeFace = cv2.CascadeClassifier(cascadePath)


font = cv2.FONT_HERSHEY_TRIPLEX

id = 0

names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

webcam = cv2.VideoCapture(0)
webcam.set(3, 640)
webcam.set(4, 480)

minWid = 0.1 * webcam.get(3)
minHei = 0.1 * webcam.get(4)

while True:
    ret, img = webcam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascadeFace.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 15, minSize = (int(minWid), int(minHei)))

    for(w, x, y, z) in faces:
        cv2.rectangle(img, (w, x), (w + y, x + z), (0, 255, 0), 2)
        id, confidence = recognize.predict(gray[x: x + z, w: w + y])

        if(confidence < 100):
            id = names[id]
            confidence = " {0}%".format(round(100 - confidence))

        else:
            id = "unknown"
            confidence = " {0}".format(round(100 - confidence))

        cv2.putText(img, str(id), (w + 5, x +5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (w + 5, x + z - 5), font, 1, (255,255,0), 1)

    cv2.imshow("webcam", img)

    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break


print("\n [INFO] Exiting program")
webcam.release()
cv2.destroyAllWindows()