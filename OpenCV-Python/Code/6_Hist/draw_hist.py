import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('picture_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# hist = cv2.calcHist([gray],[0],None,[256],[0,256]) # OpenCV 生成直方图函数
# hist = np.histogram(gray.ravel(),256,[0,256])    # Numpy 生成直方图

plt.hist(gray.ravel(), 256, [0, 256])
plt.show()
