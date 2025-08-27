import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(10, 2)

# 卷积核创建
kernel = np.ones((5, 5), np.float32)/25

while cap.isOpened() == True:
    ret, frame = cap.read()
    if ret == True:
        # 一般而言，先进行降噪操作，然后再二值化
        frame = cv2.filter2D(frame, -1, kernel)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.adaptiveThreshold(
            frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)

        cv2.imshow('res', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()
