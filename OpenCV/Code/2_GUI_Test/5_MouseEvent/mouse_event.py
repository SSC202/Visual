import cv2
import numpy as np

"""
    鼠标事件回调函数
"""
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),100,(255,0,0),1,cv2.LINE_AA)
        
        
cap = cv2.VideoCapture(1)
img = np.zeros((512,512,3),np.uint8)
cv2.namedWindow('img')
cv2.setMouseCallback('img',draw_circle)

"""
while cap.isOpened() == True:
    ret,frame = cap.read()
    if ret==True:
        cv2.imshow('img',frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
"""
    
while(1):
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()