import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("test1.jpg", 0)

# 创建 ORB 对象
orb = cv2.ORB.create()

# 寻找关键点
kp = orb.detect(img, None)

# 计算描述符
kp, des = orb.compute(img, kp)

# 绘制关键点
img2 = cv2.drawKeypoints(img, kp, None, (0, 255, 0), 0)
plt.imshow(img2), plt.show()
