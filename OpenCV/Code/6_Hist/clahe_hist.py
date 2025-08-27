"""
    局部直方图均衡化
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('picture_2.jpg', 0)
clahe = cv2.createCLAHE(8, (8, 8))
equal = clahe.apply(img)

img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
equal_hist = cv2.calcHist([equal], [0], None, [256], [0, 256])

plt.subplot(221),plt.imshow(img,'gray')
plt.subplot(222),plt.imshow(equal,'gray')
plt.subplot(223),plt.plot(img_hist),plt.plot(equal_hist)
plt.xlim([0,256])

plt.show()