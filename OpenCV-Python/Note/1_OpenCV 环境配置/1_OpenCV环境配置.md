# OpenCV Python 1_环境配置

## 1. Python 环境配置

（推荐使用Anaconda创建虚拟环境）

1. 安装Python环境或者Anaconda虚拟环境。
2. 下载 OpenCV Python 包

```shell
pip install opencv-python			# opencv-python 的基础功能包
pip install opencv-contrib-python	 # opencv-python 的基础和扩展功能包，如果上一行无法使用则使用此行
```

3. 在 VSCode 中按 F1 选择合适的解释器

![NULL](picture_1.jpg)

选择安装有opencv python包的解释器即可。

4. 测试

```python
import numpy as np
import cv2

img = cv2.imread('test.jpg', cv2.IMREAD_UNCHANGED)
cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
cv2.imshow('img', img)
cv2.waitKey(0)
```

