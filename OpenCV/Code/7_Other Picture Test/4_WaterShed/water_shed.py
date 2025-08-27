import cv2
import numpy as np

img = cv2.imread('picture.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 第一步：OTSU二值化
res, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# 第二步：图像进行开运算去除白噪声
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
# 第三步：图像距离变换
dist_transform = cv2.distanceTransform(binary, cv2.DIST_L1, 5)
# 二值化距离图像
res, sure_fg = cv2.threshold(
    dist_transform, 0.7 * (dist_transform.max()), 255, 0)
# 第四步：图像腐蚀并寻找不确定区
sure_bg = cv2.dilate(binary, kernel, iterations=3)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
# 第五步：标记
ret, marker = cv2.connectedComponents(sure_fg)
# 加一使得前景图像标志为1
marker = marker + 1
# 不确定区标记
marker[unknown == 255] = 0
# 深蓝色为未知区域，其余为浅蓝色标记
# 第六步：分水岭算法，边界标记为-1
markers = cv2.watershed(img, marker)
img[markers == -1] = [255, 0, 0]

cv2.imshow('res',img)
cv2.waitKey()
cv2.destroyAllWindows()
