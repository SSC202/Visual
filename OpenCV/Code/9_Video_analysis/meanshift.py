import cv2
import numpy as np

cap = cv2.VideoCapture('test.mp4')

# 确定窗口大小
x, y, w, h = 150, 250, 100, 100
track_window = (x, y, w, h)

# 确定颜色阈值范围
lower_red = np.array([0, 60, 32])
upper_red = np.array([180, 255, 255])

# 读取第一帧
ret, frame = cap.read()
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# 创建红色掩膜
mask = cv2.inRange(hsv_roi, lower_red, upper_red)

# 获取红色小球的直方图并归一化
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# 设置Meanshift的终止条件（最大迭代次数为10，移动小于1像素）
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 计算反向投影
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # 应用Meanshift算法
    ret, track_window = cv2.meanShift(dst, track_window, term_criteria)

    x, y, w, h = track_window
    img = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

    cv2.imshow('Meanshift Tracking', img)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
