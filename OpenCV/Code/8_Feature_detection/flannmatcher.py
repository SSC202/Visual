import numpy as np
import cv2
from matplotlib import pyplot as plt

# 读入图像
img1 = cv2.imread("test1.jpg", 0)
img2 = cv2.imread("test2.jpg", 0)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

index_params = dict(algorithm=1, tree=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# 准备一个空的掩膜来绘制好的匹配
mask_matches = [[0, 0] for i in range(len(matches))]

# 向掩膜中添加数据
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        mask_matches[i] = [1, 0]

img_matches = cv2.drawMatchesKnn(
    img1,
    kp1,
    img2,
    kp2,
    matches,
    None,
    matchColor=(0, 255, 0),
    singlePointColor=(255, 0, 0),
    matchesMask=mask_matches,
    flags=0,
)

cv2.imshow("FLANN", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
