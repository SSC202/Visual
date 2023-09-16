import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('test.jpg',0)
plt.imshow(img,cmap='gray',interpolation='bicubic')
plt.xticks([]),plt.yticks([])
plt.show()