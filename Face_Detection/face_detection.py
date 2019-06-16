# Face Recognition

# Importing the libraries
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Defining a function that will do the detections
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=2)
        '''
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]             #subzone of frame
        eyes = eye_cascade.detectMultiScale(image=roi_gray, scaleFactor=1.1, minNeighbors=3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img=roi_color, pt1=(ex, ey), pt2=(ex+ew, ey+eh), color=(0, 255, 0), thickness=2)
        '''
    return frame

'''
# Doing the Face Detection with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) or 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
'''

img = cv2.imread('imgs/7.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
new_img = detect(gray, img)
cv2.imshow('new_img', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

