import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('apple.jpg', 0)
# FFT 快速傅里叶变换
f = np.fft.fft2(img)
# 中心变换
fshift = np.fft.fftshift(f)
# fft 返回复数数组，通过取模形成波特图
magnitude_spectrum = 20*np.log(np.abs(fshift))
# 掩膜进行高通滤波，提取图像边缘信息
rows,cols = img.shape[:2]
fshift[int(rows/2)-15:int(rows/2)+15,int(cols/2)-15:int(cols/2)+15] = 0
magnitude_spectrum = 20*np.log(np.abs(fshift))
# 进行 FFT 反变换
_ifshift = np.fft.ifftshift(fshift)
_if =  np.abs(np.fft.ifft2(_ifshift))

"""
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
"""
plt.subplot(121), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Mask Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(_if, cmap='gray')
plt.title('Output Image'), plt.xticks([]), plt.yticks([])


plt.show()