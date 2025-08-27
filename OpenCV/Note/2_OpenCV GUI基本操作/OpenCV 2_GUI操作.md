# OpenCV Python 2_GUI操作

## 1. 图像的读取，显示和保存

### 图像的读取

使用`cv2.imread()`方法读入图像。

```python
"""
	读取图像函数
	第一个参数：图片路径
			图片路径应在程序工作路径或给出绝对路径
	第二个参数：读取方式
			cv2.IMREAD_COLOR 读入彩色图像 默认为1
            cv2.INREAD_GRAY_SCALE 读入灰度图像 默认为0
            cv2.IMREAD_UNCHANGED  读入图像和其ALPHA通道
"""
cv2.imread()
```

```python
import numpy as np
import cv2

img = cv2.imread('test.jpg',0)

# 如果路径错误，opencv 不会有提示，但是 img 是 None
```

### 图像的显示

使用`cv2.imshow()`方法显示图像。

```python
"""
	显示图像函数
	第一个参数：窗口名称，不同的窗口应有不同的名称
	第二个参数：图像，窗口生成后自动调整为图像大小
"""
cv2.imshow()

"""
	键盘绑定函数
	第一个参数：毫秒数
			函数等待特定的几毫秒检查是否有键盘输入，若按下键盘，则返回对应的ASCII值，程序继续运行；如果无输入则返回-1.
			若此值取0，则等待时间为无限期。
"""
cv2.waitKey()

"""
	窗口删除函数
"""
cv2.destroyAllWindows()
```

> 可以先创建窗口，再加载图像。
>
> 此时可以决定窗口是否可以调整大小。使用`cv2.namedWindow()`函数，其默认值为`cv2.WINDOW_AUTOSIZE`，调整成`cv2.WINDOW_NORMOL`时可以调整窗口大小。

### 图像的保存

使用`cv2.imwrite()`函数进行图像的保存

```python
"""
	图像保存函数
	第一个参数：图像保存的文件名
	第二个参数：图像
"""
cv2.imwrite()
```



## 2. 视频的获取和保存

### 视频的获取

视频是`VideoCapture`对象，其参数可以为摄像头设备的索引号，或者一个视频文件。

笔记本电脑的内置摄像头对应的参数就是0。可以改变此值选择别的摄像头。

```python
import cv2
import numpy as np

# 初始化 VideoCapture 对象，使用外置摄像头
cap = cv2.VideoCapture(1)

while(True):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
```

1. `cap.read()`函数返回一个元组，第一个返回值为布尔类型，用于判断是否收到视频数据。第二个值为传回的每一帧的图像。

2. 可以使用`cap.isOpened()`方法判断摄像头是否被成功初始化。若未被初始化，使用`cap.open()`函数。
3. 可以使用`cap.get(propID)`看到视频的一些信息。同时，可以使用`cap.set(propID,Value)`改变视频对应参数的值。

| 标志参数              | 值   | 含义                           |
| --------------------- | ---- | ------------------------------ |
| CAP_PROP_POS_MESC     | 0    | 视频文件的当前位置（单位为ms） |
| CAP_PROP_FRAME_WIDTH  | 3    | 视频流中图像的宽度             |
| CAP_PROP_FRAME_HEIGHT | 4    | 视频流中图像的高度             |
| CAP_PROP_FPS          | 5    | 视频流中图像帧率（每秒帧数）   |
| CAP_PROP_FOURCC       | 6    | 解编码器的4字符编码            |
| CAP_PROP_FRAME_COUNT  | 7    | 视频流中图像的帧数             |
| CAP_PROP_FORMAT       | 8    | 返回的Mat对象的格式            |
| CAP_PROP_BRIGHTNESS   | 10   | 图像的亮度（适用于相机）       |
| CAP_PROP_CONTRAST     | 11   | 图像的对比度                   |
| CAP_PROP_SATURATION   | 12   | 图像的饱和度                   |
| CAP_PROP_HUE          | 13   | 图像的色调                     |
| CAP_PROP_GAIN         | 14   | 图像的增益                     |

4. 退出程序时请使用`cap.release()`进行清理。

> 对于视频文件而言，需要将设备索引号改为文件名称。

### 视频的保存

为对每一帧图像处理后得到保存，需要创建`VideoWriter`对象

```python
"""
	VideoWriter对象创建：
	第一个参数：输出文件名
	第二个参数：编码格式
	第三个参数：播放频率
	第四个参数：帧的大小
"""
out = cv2.VideoWriter('out.avi',fourcc,20.0,(640,480))

"""
	fourcc编码对象创建：
	参数：编码格式（*'编码格式'）
"""
fourcc = cv2.VideoWriter_fourcc(*'XVID')
```

1. 编码格式

| 解编码器标志 | 含义         |
| ------------ | ------------ |
| DIVX         | MPEG-4编码   |
| PIM1         | MPEG-1编码   |
| MJPG         | JPEG编码     |
| MP42         | MPEG-4.2编码 |
| DIV3         | MPEG-4.3编码 |
| U263         | H263编码     |
| I263         | H263I编码    |
| FLV1         | FLV1编码     |

2. 使用`out.write()`函数即可将帧保存到文件中

## 3. 绘图函数

绘图统一参数：

> 1. `img`：绘图用的图像
> 2. `color`：绘图的颜色，RGB图为一个元组，灰度图给出灰度值。
> 3. `thickness`：绘图的粗细，默认为1，如果为-1则为闭合填充。
> 4. `linetype`：线条类型，默认为8连接，`cv2.LINE_AA`为抗锯齿，图像会比较平滑。

- 线条函数

```python
"""
	线条绘制函数
	第一个参数：图像
	第二个参数：左上角点
	第三个参数：右下角点
"""
cv2.line()
```

- 矩形函数

```python
"""
	矩形绘制函数
"""
cv2.rectangle()
```

- 圆形函数

```python
"""
	圆形绘制函数
	第一个参数：图像
	第二个参数：中心点坐标
	第三个参数：半径
"""
cv2.circle()
```

- 椭圆函数

```python
"""
	椭圆绘制函数
	第一个参数：图像
	第二个参数：中心点坐标
	第三个参数：两个轴的长度
	第四个参数：椭圆沿逆时针旋转的角度
	第五个参数：椭圆弧的起始角度
	第六个参数：椭圆弧的结束角度
"""
cv2.ellipse()
```

- 多边形

对于多边形，需要指定每个顶点的坐标来构建一个数组，数据类型必须为`int32`

```python
pts = np.array([[点坐标1],[点坐标2]],np.int32)
pts = pts.reshape((-1,1,2))
"""
	多边形绘制函数
	第一个参数：图片
	第二个参数：点集（用[]框住）
	第三个参数：布尔值，表示是否闭合
"""
cv2.polylines()
```

- 写文字

```python
"""
	绘制文字函数
	第一个参数：图片
	第二个参数：文字
	第三个参数：绘制位置
	第四个参数：字体
	第五个参数：字体大小
"""
cv2.putText()
```

## 4. 鼠标事件的处理

鼠标事件可以是鼠标上发生的任何动作，可以通过各种鼠标事件执行不同的任务。

通过以下代码查询鼠标事件的种类：

``` python
import cv2
events = [i for i dir(cv2) if 'EVENT' in i]
print(events)

EVENT_MOUSEMOVE              //滑动
EVENT_LBUTTONDOWN            //左键点击
EVENT_RBUTTONDOWN            //右键点击
EVENT_MBUTTONDOWN            //中键点击
EVENT_LBUTTONUP              //左键放开
EVENT_RBUTTONUP              //右键放开
EVENT_MBUTTONUP              //中键放开
EVENT_LBUTTONDBLCLK          //左键双击
EVENT_RBUTTONDBLCLK          //右键双击
EVENT_MBUTTONDBLCLK          //中键双击

EVENT_FLAG_LBUTTON        //左鍵拖曳
EVENT_FLAG_RBUTTON        //右鍵拖曳
EVENT_FLAG_MBUTTON        //中鍵拖曳
EVENT_FLAG_CTRLKEY        //(8~15)按Ctrl不放事件
EVENT_FLAG_SHIFTKEY      //(16~31)按Shift不放事件
EVENT_FLAG_ALTKEY        //(32~39)按Alt不放事件
```

鼠标事件发生后会调用对应的回调函数，从而执行对应的操作。

> 1. 创建对应的窗口`cv2.nameWindows()`；
> 2. 使用`cv2.setMouseCallback()`将窗口名和回调函数绑定；
> 3. 回调函数中，判断鼠标事件和鼠标状态以确定操作。

## 5. 滑动条

```python
"""
	滑动条创建函数：
	第一个参数：滑动条名字
	第二个参数：滑动条被放置窗口的名字
	第三个参数：滑动条默认位置
	第四个参数：滑动条的最大值
	第五个参数：回调函数
"""
cv2.createTrackbar()

"""
	滑动条键值获取函数：
	第一个参数：滑动条名字
	第二个参数：滑动条被放置窗口的名字
"""
cv2.getTrackbarPos()
```

