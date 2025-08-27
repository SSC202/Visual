"""
    直方图均衡化
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('picture_2.jpg', 0)
equal = cv2.equalizeHist(img)

img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
equal_hist = cv2.calcHist([equal], [0], None, [256], [0, 256])

# 绘制对比图像
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(equal, 'gray')
# 绘制直方图
plt.subplot(223), plt.plot(img_hist), plt.plot(equal_hist)

plt.show()
