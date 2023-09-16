import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(10, 1)

while cap.isOpened() == True:
    ret, frame = cap.read()
    if ret == True:
        # 通道分离
        # b, g, r = cv2.split(frame)
        # print(b)
        # 通道合并
        # img = cv2.merge([b, g, r])
        # cv2.imshow('img',img)
        ## 通道合并和分离耗时比较大，能使用索引操作，就使用索引操作。
        frame[:,:,1] = 0
        cv2.imshow('img',frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break

cv2.destroyAllWindows()    
cap.release()

