import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('apple_tree.jpg')
plate = cv2.imread('apple.jpg')

hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hsv_plate = cv2.cvtColor(plate,cv2.COLOR_BGR2HSV)

hist_img = cv2.calcHist([hsv_img],[0,1],None,[180,256],[0,180,0,256])
hist_plate = cv2.calcHist([hsv_plate],[0,1],None,[180,256],[0,180,0,256])

# 计算 模板/输入 进行反向投影
R = hist_plate/hist_img

h,s,v = cv2.split(hsv_img)
B = R[h.ravel(),s.ravel()]
B = np.minimum(B,1)
B = B.reshape(hsv_img.shape[:2])

# 进行卷积
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
B = cv2.filter2D(B,-1,disc)
B = np.uint8(B)
cv2.normalize(B,B,0,255,cv2.NORM_MINMAX)

ret,thresh = cv2.threshold(B,50,255,0)

while True:
    cv2.imshow('res',thresh)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
