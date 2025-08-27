import cv2
import numpy as np

img = cv2.imread('1.jpg',cv2.IMREAD_COLOR)

# 结构元素定义
# kernel = np.ones((9,9),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

while True:
    res = cv2.erode(img,kernel,iterations=10)
    cv2.imshow('res',res)
    cv2.imshow('img',img)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cv2.destroyAllWindows()
        