# OpenCV Python 3_图像基本操作

## 1. 图像信息的获取

### 图像像素的获取

使用索引的方式获取并修改像素值

```python
import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(10,1)

while cap.isOpened() == True:
    ret,frame = cap.read()
    if ret == True:
        px = frame[100,100]
        cv2.imshow('img',frame)
        print(px)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break
        
cv2.destroyAllWindows()
cap.release()
```

> 使用索引的方法并不推荐（性能低），可以使用numpy的`array.item()`和`array.itemset()`方法进行逐点修改，注意返回值为标量，需要进行分割。

### 图像属性的获取

1. `img.shape`属性

此方法可以依次返回行数，列数，通道数。

2. `img.dtype`属性

返回图像数据类型。

```python
import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(10,1)

while cap.isOpened() == True:
    ret,frame = cap.read()
    if ret == True:
        # 行数，列数，通道数（元组）
        attr = frame.shape
        print(attr)
        # 图像数据类型
        dtype = frame.dtype
        print(dtype)
        cv2.imshow('img',frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()
```

## 2. ROI

ROI，即图像的感兴趣区。用于对图像特定区域进行操作，提高操作效率。

使用索引方式获得ROI。

```python
import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(10,1)

while cap.isOpened() == True:
    ret,frame = cap.read()
    if ret == True:
		# 索引方式获取ROI
        roi = frame[100:200,100:200]
        cv2.imshow('img',roi)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break
    
cv2.destroyAllWindows()
cap.release()
```

## 3. 通道分离和合并

有时需要对R,G,B三个通道进行分离和合并操作以便于处理。

使用`img.merge()`方法对通道进行合并，输入参数为列表。

使用`img.split()`方法对通道进行拆分。

```python
import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(10, 1)

while cap.isOpened() == True:
    ret, frame = cap.read()
    if ret == True:
        # 通道分离
        # b, g, r = cv2.split(frame)
        # print(b)
        # 通道合并
        # img = cv2.merge([b, g, r])
        # cv2.imshow('img',img)
        ## 通道合并和分离耗时比较大，能使用索引操作，就使用索引操作。
        frame[:,:,1] = 0
        cv2.imshow('img',frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break

cv2.destroyAllWindows()    
cap.release()
```

## 4. 图像扩边

使用`cv2.copyMakeBorder()`对图像进行扩充边界。通常在卷积运算或0填充时被用到。

```python
"""
	图像填充函数
	第一个参数：输入图像
	第二个参数：填充类型
		1.cv2.BORDER_CONSTANT 添加有颜色的常数值边界
		2.cv2.BORDER_REFLECT 边界元素的镜像
		3.cv2.BORDER_DEFAULT
		4.cv2.BORDER_REPLICATE 重复最后一个元素
		5.cv2.BORDER_WRAP
	第三个参数：边界颜色（cv2.BORDER_CONSTANT）
"""
cv2.copyMakeBorder()
```

## 5. 图像的加运算

```python
"""
	图像加法运算
	注意：图像的大小和类型必须一致，也可以让图像和标量相加
"""
cv2.add()

"""
	图像混合运算
"""
cv2.addWeighted()
```

- 图像混合

图像混合可以体现透明效果：
$$
g(x) = (1-\alpha)f_0(x)+\alpha f_1(x)+\gamma
$$
