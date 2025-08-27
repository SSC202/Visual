import cv2
import numpy as np

img = cv2.imread('img.jpg',cv2.IMREAD_COLOR)
height,width = img.shape[:2]

pts1 = np.array([[0,0],[0,height],[width,height],[width,0]],np.float32)
pts2 = np.array([[30,30],[0,height],[width,height],[width-30,30]],np.float32)

M = cv2.getPerspectiveTransform(pts1,pts2)
res = cv2.warpPerspective(img,M,(width,height))

while True:
    cv2.imshow('res',res)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()