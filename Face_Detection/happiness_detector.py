# Happiness Detector

import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

def detect(gray, color):
    faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(color, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = color[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(image=roi_gray, scaleFactor=1.1, minNeighbors=8)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img=roi_color, pt1=(ex, ey), pt2=(ex+ew, ey+eh), color=(0, 255, 0), thickness=2)
        smiles = smile_cascade.detectMultiScale(image=roi_gray, scaleFactor=1.7, minNeighbors=22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(img=roi_color, pt1=(sx, sy), pt2=(sx+sw, sy+sh), color=(0, 0, 255), thickness=2)
    return color

oc = cv2.imread('imgs/6.jpg')
gray = cv2.cvtColor(oc, cv2.COLOR_BGR2GRAY)
new_img = detect(gray, oc)
cv2.imshow('new_img', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
        