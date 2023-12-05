import cv2

# 创建背景减除器对象
## bg_subtractor = cv2.createBackgroundSubtractorMOG2()
bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
# 打开视频文件（或者从摄像头获取）
cap = cv2.VideoCapture(1)

while True:
    # 读取当前帧
    ret, frame = cap.read()

    if not ret:
        break

    # 对当前帧应用背景减除算法
    fg_mask = bg_subtractor.apply(frame)

    # 可以对结果进行一些后处理，比如去除小的噪点
    fg_mask = cv2.medianBlur(fg_mask, 5)

    # 显示原始帧和背景减除结果
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', fg_mask)

    # 按 'q' 键退出循环
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
