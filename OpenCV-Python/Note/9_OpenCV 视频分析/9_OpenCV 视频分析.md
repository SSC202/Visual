# OpenCV-Python 9_视频分析

## 1. 概率跟踪方法

### Meanshift 算法

[(参考文章)](https://blog.csdn.net/lk3030/article/details/84108765)

Meanshift的本质是一个迭代的过程，在一组数据的密度分布中，使用无参密度估计寻找到局部极值（不需要事先知道样本数据的概率密度分布函数，完全依靠对样本点的计算）。

在d维空间中，任选一个点，然后以这个点为圆心，h为半径做一个高维球，因为有d维，d可能大于2，所以是高维球。落在这个球内的所有点和圆心都会产生一个向量，向量是以圆心为起点落在球内的点位终点。然后把这些向量都相加。相加的结果就是下图中黄色箭头表示的Meanshift向量。

![NULL](picture_1.jpg)

然后，再以这个Meanshift 向量的终点为圆心，继续上述过程，又可以得到一个Meanshift 向量。

不断地重复这样的过程，可以得到一系列连续的Meanshift 向量，这些向量首尾相连，最终可以收敛到概率密度最大的点。

Meanshift算法的本质是从起点开始，一步步迭代从而到达样本特征点的密度中心。

> 由于图像经过直方图反向投影后得到的为概率图像，通过颜色直方图反向投影可以方便的进行最大概率处的跟踪，从而实现图像的跟踪。

```python
"""
	meanShift 算法函数
	probImage:		输入的直方图反向投影图像
	window:			搜索窗口的初始位置和大小
	criteria:		Meanshift算法的终止条件
"""
retval, window = cv2.meanShift(probImage, window, criteria)


## criteria是一个包含三个元素的元组 (type, maxCount, epsilon)
## type: 终止条件的类型。可以是 cv2.TERM_CRITERIA_EPS 表示按照指定精度停止，或者 cv2.TERM_CRITERIA_COUNT 表示按照迭代次数停止，或者两者的组合 cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT 表示两者之一满足即可终止。
## maxCount: 最大迭代次数，即算法进行迭代的最大次数。
## epsilon: 指定的精度，用于判断是否达到终止条件。
```

```python
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
```

> 1. **优点**
>
> - 算法计算量不大，在目标区域已知的情况下完全可以做到实时跟踪；
>
> - 采用核函数直方图模型，对边缘遮挡、目标旋转、变形和背景运动不敏感。
>
> 2. **缺点**
>
> - 跟踪过程中由于窗口宽度大小保持不变，框出的区域不会随着目标的扩大（或缩小）而扩大（或缩小）；
>
> - 当目标速度较快时，跟踪效果不好；
>
> - 直方图特征在目标颜色特征描述方面略显匮乏，缺少空间信息；
> - 需要指定初始窗口。

### Camshift算法

Camshift是连续的自适应Meanshift算法，可以在跟踪的过程中随着目标大小的变化实时调整搜索窗口大小，对于视频序列中的每一帧还是采用Meanshift来寻找最优迭代结果。

在Meanshift算法中寻找搜索窗口的质心用到窗口的零阶矩$M_{00}$和一阶矩$M_{10}$，$M_{01}$，零阶矩是搜索窗口内所有像素的积分，即所有像素值之和，物理上的意义是计算搜索窗口的尺寸。经过目标的直方图反向投影后，目标区域的搜索窗口大部分像素值归一化后应该是最大值255，如果计算出来零阶矩大于某一阈值，可以认为此时目标铺满了整个搜索窗口，有理由认为在搜索窗口之外的区域还存在目标区域，需要增大搜索窗口的尺寸；相应的，如果零阶矩小于某一阈值，则需要缩小搜索窗口的尺寸，如此一来，当目标的大小发生变化的时候，Camshift算法就可以自适应的调整目标区域进行跟踪。
```python
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

# 设置Camshift的终止条件（最大迭代次数为10，移动小于1像素）
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 计算反向投影
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # 应用Camshift算法
    res, track_window = cv2.CamShift(dst, track_window, term_criteria)

    pts = cv2.boxPoints(res)
    pts = np.int0(pts)
    img = cv2.polylines(frame, [pts], True, 255, 2)

    cv2.imshow('Meanshift Tracking', img)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

