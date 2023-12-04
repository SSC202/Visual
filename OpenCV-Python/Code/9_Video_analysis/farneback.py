import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取视频文件
cap = cv2.VideoCapture('test.mp4')

# 读取第一帧
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# 创建一个图像，用于显示光流场
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    # 读取新的一帧
    ret, frame2 = cap.read()
    if not ret:
        break
    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 计算Farneback光流
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 极坐标转换
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 将角度映射到HSV色彩空间
    hsv[..., 0] = ang * 180 / np.pi / 2

    # 将光流的幅度映射到HSV的亮度通道
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # 将HSV图像转回BGR
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 显示结果
    cv2.imshow('Optical Flow Analysis', bgr)

    # 更新上一帧图像
    prvs = next_frame

    # 按下ESC键退出
    if cv2.waitKey(30) & 0xFF == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
