import cv2
import os

web_cam = cv2.VideoCapture(0)
web_cam.set(3, 640) #width
web_cam.set(4, 480) #height

detect_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_id = input("\n Enter user ID and hit Enter")

print("\n [INFO] Initializing face capture.")

a = 0
while(True):
    ret, img = web_cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_face.detectMultiScale(gray, 1.3, 5)

    for(w, x, y, z) in faces:
        cv2.rectangle(img, (w, x), (w + y, x + z), (255, 0, 0), 2)
        a+=1

        cv2.imwrite("data/User."+str(face_id)+"."+str(a)+".jpg", gray[x:x+z, w:w+y])
        cv2.imshow("image", img)

    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break

    elif a >= 30:
        break


print("\n [INFO] Exiting program")
web_cam.release()
cv2.destroyAllWindows()