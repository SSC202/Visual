import cv2
import numpy as np

# 读取视频文件
cap = cv2.VideoCapture(1)

# 读取第一帧
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# 设置角点初始位置
p0 = cv2.goodFeaturesToTrack(
    old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# 图像金字塔参数
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 创建一个掩膜用于绘制轨迹
mask = np.zeros_like(old_frame)

while True:
    # 读取新的一帧
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 计算光流
    p1, status, err = cv2.calcOpticalFlowPyrLK(
        old_gray, frame_gray, p0, None, **lk_params)

    # 仅保留跟踪成功的点
    if p1 is not None:
        good_new = p1[status == 1]
        good_old = p0[status == 1]
    else:
        pass

    # 绘制轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)),
                        (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

    # 合并原始图像和轨迹
    img = cv2.add(frame, mask)

    # 显示结果
    cv2.imshow('Optical Flow Tracking', img)

    # 更新角点位置
    p0 = good_new.reshape(-1, 1, 2)

    # 按下ESC键退出
    if cv2.waitKey(30) & 0xFF == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
