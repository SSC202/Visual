import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(10,2)

while cap.isOpened() == True:
    ret,frame = cap.read()
    if ret == True:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # sobel 算子
        # 1,0 x方向
        sobel = cv2.Sobel(frame,-1,1,0,ksize = 5)
        # 0,1 y方向
        sobel = cv2.Sobel(sobel,-1,0,1,ksize = 5)
        # laplacian 算子
        laplacian = cv2.Laplacian(frame,-1)
        cv2.imshow('sobel',sobel)
        cv2.imshow('laplacian',laplacian)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break
    
cv2.destroyAllWindows()
cap.release()
        