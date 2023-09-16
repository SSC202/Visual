import cv2
import numpy as np

i = 1

img1 = cv2.imread('1.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('2.jpg', cv2.IMREAD_COLOR)

ret1 = img1[0:50, 0:50]
ret2 = img2[0:50, 0:50]

while True:
    # ret = cv2.add(ret1,ret2)
    ret = cv2.addWeighted(ret1, i, ret2, (1-i), 0)
    i = i - 0.01
    cv2.imshow('img', ret)
    if i == 0:
        break


cv2.destroyAllWindows()
