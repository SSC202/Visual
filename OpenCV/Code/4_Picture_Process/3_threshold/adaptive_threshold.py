import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(10,2)

while cap.isOpened() == True:
    ret,frame = cap.read()
    if ret == True:
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # 自适应阈值
        res = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,-10)
        cv2.imshow('res',res)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()