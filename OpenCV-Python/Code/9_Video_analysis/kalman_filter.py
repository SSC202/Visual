import cv2
import numpy as np

frame = np.zeros((800, 800, 3), np.uint8)

last_measurement = current_measurement = np.array((2, 1), np.float32)
last_predicition = current_prediction = np.zeros((2, 1), np.float32)

"""
    传递X,Y的坐标值,便于对轨迹进行卡尔曼滤波
"""


def mousemove(event, x, y, s, p):
    global frame, current_measurement, measurements, last_measurement, current_prediction, last_prediction
    last_measurement = current_measurement
    last_prediction = current_prediction
    # 传递当前测量坐标值
    current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
    # 修正卡尔曼滤波的预测结果
    kalman.correct(current_measurement)
    # 调用Kalman这个类的predict方法得到状态的预测值矩阵,用来估算目标位置
    current_prediction = kalman.predict()
    # 上一次测量值
    lmx, lmy = last_measurement[0], last_measurement[1]
    # 当前测量值
    cmx, cmy = current_measurement[0], current_measurement[1]
    # 上一次预测值
    lpx, lpy = last_prediction[0], last_prediction[1]
    # 当前预测值
    cpx, cpy = current_prediction[0], current_prediction[1]

    cv2.line(frame, (int(lmx), int(lmy)), (int(cmx), int(cmy)), (0, 100, 0))
    cv2.line(frame, (int(lpx), int(lpy)), (int(cpx), int(cpy)), (0, 0, 200))


cv2.namedWindow("kalman_tracker")

cv2.setMouseCallback("kalman_tracker", mousemove)

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array(
    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
)
kalman.processNoiseCov = (
    np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    * 0.03
)

while True:
    cv2.imshow("kalman_tracker", frame)
    if (cv2.waitKey(30) & 0xFF) == 27:
        break

cv2.destroyAllWindows()
