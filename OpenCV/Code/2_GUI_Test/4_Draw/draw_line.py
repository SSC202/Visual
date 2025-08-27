import numpy as np
import cv2

cap = cv2.VideoCapture(1)

cap.set(10,1)

while(cap.isOpened() == True):
    ret,frame = cap.read()
    if(ret == True):
        cv2.line(frame,(0,0),(640,480),(255,0,0),3)
        cv2.rectangle(frame,(300,200),(340,280),(0,255,0),2)
        cv2.circle(frame,(320,240),32,(0,0,255),1,cv2.LINE_AA)
        cv2.ellipse(frame,(320,240),(100,50),0,0,180,(0,255,255),1,cv2.LINE_AA)
        
        pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(frame,[pts],True,(255,0,0),2,cv2.LINE_AA)
        
        cv2.putText(frame,'OpenCV',(10,300),cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,0),1,cv2.LINE_AA)
        
        cv2.imshow('capture',frame)
        key = cv2.waitKey(1)
        if(key == ord('q')):
            break

cap.release()
cv2.destroyAllWindows()

        

