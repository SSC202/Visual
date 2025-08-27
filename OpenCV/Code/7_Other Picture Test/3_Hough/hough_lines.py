import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(10, 2)

while cap.isOpened() == True:
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = cv2.Canny(gray, 50, 150, apertureSize=3)

        # 霍夫直线变换
        lines = cv2.HoughLines(binary, 1, np.pi/180, 200)
        
        if not lines is None != False: # 判断列表是空列表还是NoneType类型列表，避免无法遍历
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x_0 = a*rho
                y_0 = b*rho
                x1 = int(x_0+1000*(-b))
                y1 = int(y_0+1000*a)
                x2 = int(x_0-1000*(-b))
                y2 = int(y_0-1000*a)

                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow('res', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()
