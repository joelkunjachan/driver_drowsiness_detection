import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer #Pygame is used for handling and playing audio files.
import time #to perform tasks such as measuring the execution time of code, adding delays, getting the current time, and more.


mixer.init() #used to initialize the Pygame mixer. It must be called before you can start working with audio files and playing sounds.
sound = mixer.Sound("D:\nextgen_projects\projects_akhil&anju_now\Drowsiness Detection (2)\ Detection\Drowsiness Detection\ML\alarm.wav")

face = cv2.CascadeClassifier('D:\\nextgen_projects\\projects_akhil&anju_now\\Drowsiness Detection (2)\\Drowsiness Detection\\Drowsiness Detection\\ML\\haar cascade files\\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('D:\\nextgen_projects\\projects_akhil&anju_now\\Drowsiness Detection (2)\\Drowsiness Detection\\Drowsiness Detection\\ML\\haar cascade files\\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('D:\\nextgen_projects\\projects_akhil&anju_now\\Drowsiness Detection (2)\\Drowsiness Detection\\Drowsiness Detection\\ML\\haar cascade files\\haarcascade_righteye_2splits.xml')



lbl=['Close','Open']

model = load_model('D:\\nextgen_projects\\projects_akhil&anju_now\\Drowsiness Detection (2)\\Drowsiness Detection\\Drowsiness Detection\\ML\\\\models\\cnnCat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25)) #used in OpenCV for object detection.
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED ) # drawing a filled rectangle on the frame image using OpenCV.

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w] ## Extracts the region of interest (ROI) from the frame image using the coordinates of the right eye region. It creates a new image r_eye that contains only the right eye region.
        count=count+1 #This is likely used to keep track of the number of eye regions processed.
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24)) #This ensures that the input to the model is of a consistent size.
        r_eye= r_eye/255 # which is often done to improve the performance of neural networks
        r_eye=  r_eye.reshape(24,24,-1) #Reshapes the 2D grayscale image r_eye to a 3D array with shape (24, 24, 1). The additional dimension of size 1 is added to match the input shape expected by the model.
        r_eye = np.expand_dims(r_eye,axis=0) #Adds an extra dimension to r_eye at the 0th axis to create a 4D array. This is necessary to match the input shape expected by the model.
        rpred = model.predict_classes(r_eye) #
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w] 
        count=count+1 #
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0): #indicating that both eyes are classified as closed. If this condition is true, the following actions are performed:
        score=score+1 
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA) #The text "Closed" is displayed on the frame image at the specified position using the cv2.putText() function. This text is used to indicate that the eyes are closed.
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>15):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
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
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
