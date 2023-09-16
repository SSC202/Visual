# OpenCV 14_图像矩

矩是描述图像特征的算子，被广泛用于图像检索和识别、图像匹配、图像重建、图像压缩以及运动图像序列分析等领域。

- 几何矩
$$ 
m_{ji} = \Sigma_{x,y}I(x,y)x^jy^i
$$

其中I(x,y)是像素(x,y)处的像素值。当x和y同时取值0时称为零阶矩，零阶矩可以用于计算某个形状的质心，当x和y分别取值0和1时被称为一阶矩。

- 中心矩
$$
mu_{ji} = \Sigma_{x,y}I(x,y)(x-\overline x)^j(y - \overline y)^i
$$

归一化几何矩
$$
nu_{ji} = \frac{mu_{ji}}{m_{00}^{(i+j)/2+1}}
$$
```c++
Moments cv::moments(InputArray  array,bool  binaryImage = false );
```
>array：计算矩的区域2D像素坐标集合或者单通道的CV_8U图像
>
>binaryImage：是否将所有非0像素值视为1的标志。

该函数用于计算图像连通域的几何矩和中心距以及归一化的几何矩。函数第一个参数是待计算矩的输入图像或者2D坐标集合。函数第二个参数为是否将所有非0像素值视为1的标志，该标志只在第一个参数输入为图像类型的数据时才会有作用。

函数会返回一个Moments类的变量，Moments类中含有几何矩、中心距以及归一化的几何矩的数值属性

>moments 种类 | 属性
>-|-
>spatial monents|m00,m10,m01,m20,m11,m02,m30,m21,m12,m03
>central moments|mu20,mu11,mu02,mu30,mu21,mu12,mu03
>central normalized moments|nu20,nu11,nu02,nu30,nu21,nu12,nu03

- Hu矩

Hu矩具有旋转、平移和缩放不变性，因此在图像具有旋转和放缩的情况下Hu矩具有更广泛的应用领域。Hu矩是由二阶和三阶中心距计算得到七个不变矩.

![NULL](picture_1.jpg)
```c++
void cv::HuMoments(const Moments &  moments,
                   double  hu[7] 
                  );
void cv::HuMoments(const Moments &  m,
                   OutputArray  hu 
                  );
```

>moments：输入的图像矩
>
>hu[7]：输出Hu矩的七个值
>
>m：输入的图像矩
>
>hu：输出Hu矩的矩阵

- Hu矩轮廓匹配
```c++
double cv::matchShapes(InputArray  contour1,
                       InputArray  contour2,
                       int  method,
                       double  parameter 
                      );
```
>contour1：原灰度图像或者轮廓
>
>contour2：模板图像或者轮廓
>
>method：匹配方法的标志。
>
>parameter：特定于方法的参数