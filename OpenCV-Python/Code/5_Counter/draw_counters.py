import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(10, -1)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

while cap.isOpened() == True:
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        img = cv2.morphologyEx(binary, cv2.MORPH_ERODE,kernel)
        # 寻找轮廓
        contours, hierarchy = cv2.findContours(
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 在原图中画出轮廓
        cv2.drawContours(frame, contours, -1, (0, 255, 255), 3)
        cv2.imshow('img', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()
