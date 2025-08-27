import cv2
import math
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(10, 2)
alpha = 45

while cap.isOpened() == True:
    ret, frame = cap.read()
    if ret == True:
        height, width = frame.shape[:2]
        x_0, y_0 = width/2, height/2
        # 旋转后窗口大小计算
        new_height = (height*math.cos(alpha/180 * math.pi)) + \
            (width*math.sin(alpha/180 * math.pi))
        new_width = (width*math.cos(alpha/180 * math.pi)) + \
            (height*math.sin(alpha/180 * math.pi))
        # 平移
        N = np.zeros((2, 3), np.float32)
        N[0, 0] = 1
        N[1, 1] = 1
        N[0, 2] = new_width/2 - x_0
        N[1, 2] = new_height/2 - y_0
        img = cv2.warpAffine(frame, N, (int(new_width), int(new_height)))
        # 构建旋转矩阵并进行旋转
        M = cv2.getRotationMatrix2D(
            (int(new_width)/2, int(new_height)/2), alpha, 1.0)
        res = cv2.warpAffine(img, M, (int(new_width), int(new_height)))

        cv2.imshow('res', res)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()
