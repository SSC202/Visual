import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('apple_tree.jpg')
plate = cv2.imread('apple.jpg')

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)

hist_plate = cv2.calcHist([hsv_plate], [0, 1], None, [
                          180, 256], [0, 180, 0, 256])

# 归一化直方图
cv2.normalize(hist_plate, hist_plate, 0, 255, cv2.NORM_MINMAX)
# 反向投影
dst = cv2.calcBackProject([hsv_img], [0, 1], hist_plate, [0, 180, 0, 256], 1)

# 卷积用来连接分散的点
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dst = cv2.filter2D(dst, -1, disc)


# 二值化概率图像
ret, thresh = cv2.threshold(dst, 50, 255, 0)
thresh = cv2.merge((thresh, thresh, thresh))

# 按位操作进行掩膜计算
res = cv2.bitwise_and(img, thresh)
res = np.hstack((img, thresh, res))

while True:
    cv2.imshow('res', res)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
