# OpenCV 6_图像直方图（Hist）

直方图用于**分析整体图像的灰度分布**，注意，使用直方图分析前，请将图像转换为灰度图。

## 1. 直方图的绘制

### 直方图的基本属性

`BINS`：直方图的小组数量，会将256个像素值划分为`BINS`个小组。

`DIMS`：收集数据的参数个数，设置为1；

`RANGE`：统计灰度值范围。

### 直方图的绘制

```python
"""
	直方图生成函数
	第一个参数：原图像（uint8或float32类型），列表
	第二个参数：统计直方图的图像类型，灰度图填[0],彩色图填[0]/[1]/[2]对应BGR；
	第三个参数：掩膜图像，如果为整个图像为None
	第四个参数：BIN的数目，为列表
	第五个参数：像素值的范围
	返回值：直方图
"""
cv2.calcHist()

"""
	numpy 直方图生成函数
	第一个参数：一维图像数组，使用img.ravel()函数
	第二个参数：小组数
	第三个参数：统计像素值范围
"""
np.histogram()

"""
	matplotlib 直方图绘制生成函数
	第一个参数：一维图像数组，使用img.ravel()函数
	第二个参数：小组数
	第三个参数：统计像素值范围
"""
plt.hist()
```

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('picture_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# hist = cv2.calcHist([gray],[0],None,[256],[0,256]) # OpenCV 生成直方图函数
# hist = np.histogram(gray.ravel(),256,[0,256])    # Numpy 生成直方图

plt.hist(gray.ravel(), 256, [0, 256])
plt.show()
```

### 掩膜的设置

将要计算灰度分布的部分设置为白色，不要的部分设置为黑色，即可实现掩膜。

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('picture_1.jpg', 0)
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:300] = 255

mask_img = cv2.bitwise_and(img, img, mask=mask)

hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(mask_img, 'gray')
plt.subplot(224), plt.plot(hist_mask),plt.plot(hist_full)
plt.xlim([0,256])

plt.show()
```

## 2. 直方图均衡化

### 全局直方图均衡化

对于像素集中分布在一个区间内的图像，通常使用直方图均衡化增强对比度，使得每个灰度都能够在$x$轴上均匀分布，提高图像的质量，这就是直方图均衡化。

直方图均衡化的步骤如下：

1. 统计原始图像的直方图。计算每个像素值出现的次数，然后将它们归一化，得到每个像素值的频率。

2. 计算累积分布函数（CDF）

   CDF 是对频率分布函数（PDF）的积分，它表示每个像素值在原始图像中出现的概率。CDF 可以通过对 PDF 进行累加计算得到。对于一个灰度值$i$，CDF 的计算公式如下：

$$
CDF(i) = \sum^i_{j = 0}P(j)
$$

$P(j)$为灰度值为$j$的像素在图像中出现的频率。

3. 计算均衡化后的像素值。需要将原始图像中的每个像素值映射到一个新的像素值，使得均衡化后的直方图近似为一个均匀分布的直方图。

$$
H(i) = round(\frac{CDF(i) - CDF(min)}{(M \times N - 1)}\times(L-1))
$$

其中，$H(i)$表示映射后的像素值，$M$和$N$分别表示图像的宽度和高度，$L$表示像素值的范围，$min$表示原始图像中的最小像素值。

4. 将原始图像中的像素值替换为映射后的像素值。

```python
"""
	全局直方图均衡化函数
	参数：灰度图
"""
cv2.equalizeHist()
```

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('picture_2.jpg', 0)
equal = cv2.equalizeHist(img)

img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
equal_hist = cv2.calcHist([equal], [0], None, [256], [0, 256])

# 绘制对比图像
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(equal, 'gray')
# 绘制直方图
plt.subplot(223), plt.plot(img_hist), plt.plot(equal_hist)

plt.show()
```

> 全局直方图均衡化的缺陷
>
> 1. 全局变换：直方图均衡化是一种全局变换方法，它将整个图像的直方图都变成了均匀分布的直方图，这可能会导致一些像素值的细节信息丢失或被模糊化。
>
> 2. 非线性变换：直方图均衡化的映射函数是非线性的，这意味着它会改变像素值之间的距离，从而可能导致一些图像特征的失真。
>
> 3. 计算复杂度：直方图均衡化需要计算原始图像的直方图和累计分布函数，这可能会增加算法的计算复杂度，尤其是对于大型图像。
>
> 4. 对噪声敏感：由于直方图均衡化是一种全局变换，它对图像中的噪声也会进行增强，可能会使噪声更加明显。

### 限制对比度自适应直方图均衡化（CLAHE）

**基本原理**：

1. 将原始图像分成许多小块（或称为子图像），每个小块大小为 $N \times N$。

2. 对于每个小块，计算其直方图，并将直方图进行均衡化，得到映射函数。

3. 对于每个小块，使用对应的映射函数对其像素值进行变换。

4. 由于像素值在小块之间可能存在不连续的变化，因此需要进行双线性插值处理，以使得整个图像的对比度保持连续。

> CLAHE 的优点：
>
> 1. 保留了图像的局部细节信息，不会将整个图像都变成均匀分布的直方图。
> 2. 具有自适应性，可以根据图像的局部特征来调整直方图均衡化的参数。
> 3. 通过限制对比度，可以有效地减少直方图均衡化算法对噪声的敏感性。

```python
"""
	CLAHE 模板生成函数
	第一个参数 clipLimit：颜色对比度的阈值，可选项，默认8
	第二个参数 titleGridSize：局部直方图均衡化的模板（邻域）大小，可选项，默认值 (8,8)
	返回值：CLAHE 对象
"""
cv2.createCLAHE()

"""
	CLAHE 处理函数
	参数：灰度图
"""
clahe.apply()
```

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('picture_2.jpg', 0)
clahe = cv2.createCLAHE(8, (8, 8))
equal = clahe.apply(img)

img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
equal_hist = cv2.calcHist([equal], [0], None, [256], [0, 256])

plt.subplot(221),plt.imshow(img,'gray')
plt.subplot(222),plt.imshow(equal,'gray')
plt.subplot(223),plt.plot(img_hist),plt.plot(equal_hist)
plt.xlim([0,256])

plt.show()
```

