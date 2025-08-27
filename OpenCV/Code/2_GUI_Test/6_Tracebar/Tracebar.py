import numpy as np
import cv2

img = np.zeros((512,512,3),np.uint8)
cv2.namedWindow('image')

"""
    滑动条回调函数
"""
def trackbar_callback():
    pass

cv2.createTrackbar('R','image',0,255,trackbar_callback)
cv2.createTrackbar('G','image',0,255,trackbar_callback)
cv2.createTrackbar('B','image',0,255,trackbar_callback)

switch = '0:OFF/1:ON'
cv2.createTrackbar(switch,'image',0,1,trackbar_callback)

while(1):
    cv2.imshow('image',img)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')
    s = cv2.getTrackbarPos(switch,'image')
    
    if(s == 0):
        img[:] = 0
    else:
        img[:] = [b,g,r]
        
cv2.destroyAllWindows()