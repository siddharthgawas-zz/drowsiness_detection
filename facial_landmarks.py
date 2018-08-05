import numpy  as np
import cv2
import dlib
from imutils import face_utils
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects = detector(gray,0)
    for rect in rects:
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)
        for (x,y) in shape:
            cv2.circle(frame,(x,y),1,(0,0,255),-1)

    cv2.imshow('video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()