import cv2
import os
import numpy as np
from pygame import mixer
import time
###
import dlib
from imutils import face_utils





#####

YAWN_THRESH = 20
EAR_THRESH = 0.25  # Eye Aspect Ratio threshold for closed eyes

def eye_aspect_ratio(eye):
    """Calculate the eye aspect ratio (EAR) to detect if eyes are closed"""
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


mixer.init()
sound = mixer.Sound("ml\\alarm.wav")

# Initializing the dlib face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('ml\\shape_predictor_68_face_landmarks.dat')

# Grabbing the indexes of the facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]



lbl=['Close','Open']
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
eye_closed_frames=0  # Counter for consecutive frames with closed eyes

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2]
    yawn_status=False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting faces in the grayscale frame
    rects = detector(gray, 0)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    # Looping over the face detections
    for rect in rects:
        # Determining the facial landmarks for the face region
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extracting the left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        # Computing the eye aspect ratio for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        # Averaging the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        
        # Computing the convex hull for the left and right eye, then visualizing
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # Checking if the eye aspect ratio is below the threshold
        if ear < EAR_THRESH:
            eye_closed_frames += 1
        else:
            eye_closed_frames = 0
        
        # Extract lips for yawn detection
        #distance = lip_distance(shape)
        # if (distance > YAWN_THRESH):
        #     cv2.putText(frame, "Yawn Alert", (10, 30),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #     yawn_status=True
        # else:
        #     yawn_status=False

    # Update drowsiness score based on eye closure
    if eye_closed_frames > 0:
        score=score+0.5
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>15 or yawn_status):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        if score>30:
            try:
                sound.play()
            except:  # isplaying = False
                pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
        cv2.putText(frame, "Sleep Alert!! Take Rest", (10, 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
