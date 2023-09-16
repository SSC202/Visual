import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('picture_1.jpg', 0)
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:300] = 255

mask_img = cv2.bitwise_and(img, img, mask=mask)

hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(mask_img, 'gray')
plt.subplot(224), plt.plot(hist_mask),plt.plot(hist_full)
plt.xlim([0,256])

plt.show()