import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread("test1.jpg", 0)
img2 = cv2.imread("test2.jpg", 0)

orb = cv2.ORB.create()

# 关键点检测
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 创建 BF 匹配器
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 进行匹配
matches = bf.match(des1, des2)

# 按照距离排序
matches = sorted(matches, key=lambda x: x.distance)

# 绘制匹配结果
img3 = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    matches[:10],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

plt.imshow(img3)
plt.show()
