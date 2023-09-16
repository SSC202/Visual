import cv2
import numpy as np

img = cv2.imread('img.jpg',cv2.IMREAD_COLOR)
height,width = img.shape[:2]

pts1 = np.array([[50,50],[100,20],[20,100]],np.float32)
pts2 = np.array([[10,10],[90,5],[5,90]],np.float32)

M = cv2.getAffineTransform(pts1,pts2)
res = cv2.warpAffine(img,M,(width,height))

while True:
    cv2.imshow('res',res)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()