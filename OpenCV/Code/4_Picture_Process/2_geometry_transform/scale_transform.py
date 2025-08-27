import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(10,2)


while cap.isOpened() == True:
    ret,frame = cap.read()
    if ret == True:
        # 图像使用两倍放大的放大因子
        # img = cv2.resize(frame,None,fx = 2,fy = 2,interpolation=cv2.INTER_LINEAR)
        # 图像指定尺寸缩放
        height,width = frame.shape[:2]
        img = cv2.resize(frame,(2*width,2*height),interpolation=cv2.INTER_LINEAR)
        cv2.imshow('img',img)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()