import cv2
import numpy as np

orange_img = cv2.imread('orange.jpg',cv2.IMREAD_COLOR)
apple_img = cv2.imread('apple.jpg',cv2.IMREAD_COLOR)

# 创建apple的高斯金字塔，下采样
G = apple_img.copy()
pyr_apple = [G]
for i in np.arange(4):
    G = cv2.pyrDown(G)
    pyr_apple.append(G)

# 创建orange的高斯金字塔，下采样
G = orange_img.copy()
pyr_orange = [G]
for i in np.arange(4):
    G = cv2.pyrDown(G)
    pyr_orange.append(G)

# 创建apple的拉普拉斯金字塔
lap_apple = [pyr_apple[3]]
for i in range(3, 0, -1):
    GE = cv2.pyrUp(pyr_apple[i])
    L = cv2.subtract(pyr_apple[i-1], GE)
    lap_apple.append(L)

# 创建orange的拉普拉斯金字塔
lap_orange = [pyr_orange[3]]
for i in range(3, 0, -1):
    GE = cv2.pyrUp(pyr_orange[i])
    L = cv2.subtract(pyr_orange[i-1], GE)
    lap_orange.append(L)

# 将两个图像的矩阵的左半部分和右半部分拼接到一起
LS = []
for la, lb in zip(lap_apple, lap_orange):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, 0:cols//2], lb[:, cols//2:]))
    LS.append(ls)


ls_ = LS[0]  # 这里LS[0]为高斯金字塔的最小图片
for i in range(1, 4):  # 第一次循环的图像为高斯金字塔的最小图片，依次通过拉普拉斯金字塔恢复到大图像
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])  # 采用金字塔拼接方法的图像

while True:
    cv2.imshow('res', ls_)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
