import cv2
import numpy as np

cap = cv2.VideoCapture(1)

if cap.isOpened() == False:
    cap.open()

# 视频的基础设置
wide = cap.get(3)
height = cap.get(4)
bright = cap.set(10, 1)

# VideoWriter 对象创建
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('out.avi',fourcc,20.0,(640,480))

while (True):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame, 0)
        # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        cv2.imshow('video',frame)
        out.write(frame)
        print("%d,%d,%d" % (wide, height, bright))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
