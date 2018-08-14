import numpy  as np
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance
import time
from imutils.video import VideoStream
from threading import Thread
import imutils

def get_ear(eye):
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    ear = (a+b)/(2.0*c)
    return ear


# Threshold of Eye aspect ratio which detects eye closed position
EAR_THRESHOLD = 0.30
# Number of consecutive frames eyes are closed which detects drowsiness
EAR_CONSECUTIVE_FRAMES = 40
COUNTER = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Indices of landmarks of eyes.
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

video_stream = VideoStream(src=0).start()
time.sleep(1.0)

while True:
    frame = video_stream.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = get_ear(leftEye)
        rightEAR = get_ear(rightEye)
        ear_avg = (rightEAR+leftEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame,[leftEyeHull],-1,(255,0,0),1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 1)

        if ear_avg < EAR_THRESHOLD:
            COUNTER += 1
            if COUNTER >= EAR_CONSECUTIVE_FRAMES:
                cv2.putText(frame,"DROWSY!!",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        else:
            COUNTER = 0
        cv2.putText(frame,"EAR = "+str(ear_avg),(300,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    if key==ord('q'):
        break

cv2.destroyAllWindows()
video_stream.stop()