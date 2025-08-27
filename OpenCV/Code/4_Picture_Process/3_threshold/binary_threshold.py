import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(10, 2)

while cap.isOpened() == True:
    ret, frame = cap.read()
    if ret == True:
        # 首先转换为灰度图
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # 简单阈值（正向）
        ## retval, res = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # 简单阈值（反向）
        retval,res = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow('res', res)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()
