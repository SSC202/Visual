import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('picture_1.jpg')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hist = cv2.calcHist([hsv_img], [0, 1], None, [180, 256], [0, 180, 0, 256])

plt.imshow(hist, interpolation='nearest')

plt.show()
