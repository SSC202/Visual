# OpenCV 5_图像变换

## 1. 仿射变换

实现图像的旋转首先需要确定旋转角度和旋转中心，之后 确定旋转矩阵，最终通过仿射变换实现图像旋转。针对这个流程，OpenCV提供了`getRotationMatrix2D()`函数用于计算旋转矩阵和`warpAffine()`函数用于实现图像的仿射变换。

1. 计算旋转矩阵函数
```c++
Mat cv::getRotationMatrix2D (Point2f center,double  angle,double  scale)
```
>center：图像旋转的中心位置。
>
>angle：图像旋转的角度，单位为度，正值为逆时针旋转。
>
>scale：两个轴的比例因子，可以实现旋转过程中的图像缩放，不缩放输入1。

旋转矩阵与旋转角度和旋转中心的关系:
$$
Rotation = 
\left[
\begin{matrix}
\alpha & \beta & (1-\alpha)*center.x-\beta*center.y \\
-\beta & \alpha & \beta*center.x+(1-\alpha)*center.y
\end{matrix}
\right]
$$
$$
\alpha = scale*cos(angle)
$$
$$
\beta = scale*sin(angle)
$$

2. 仿射变换函数
```c++
void cv::warpAffine(InputArray src,
                    OutputArray dst,
                    InputArray M,
                    Size dsize,
                    int  flags = INTER_LINEAR,
                    int  borderMode = BORDER_CONSTANT,
                    const Scalar& borderValue = Scalar()
                   );
```
>src：输入图像。
>
>dst：仿射变换后输出图像，与src数据类型相同，但是尺寸与dsize相同。
>
>M：2×3的变换矩阵。
>
>dsize：输出图像的尺寸。
>
>flags：插值方法标志。
>
>borderMode：像素边界外推方法的标志。
>
>borderValue：填充边界使用的数值，默认情况下为0。

>flags|值|含义
>-|-|-
>WARP_FILL_OUTLIERS|8|填充所有输出图像的像素，如果部分像素落在输入图像的边界外，那么他们的值设定为fillval。
>WARP_INVERSE_MAP|16|表示M为输出图像到输入图像的反变换

>borderMode|值|含义
>-|-|-
>BORDER_CONSTANT|0|用特定值填充，如iiiiii\|abcdefh\|iiiiii
>BORDER_PRELICATE|1|两端复制填充，如aaaaaa\|abcdefgh\|hhhhhhh
>BORDER_REFLECT|2|倒序填充，如fedcba\|abcdefgh\|hgfedcb
>BORDER_WARP|3|正序填充，如cdefgh\|abcdefgh\|abcdefg
>BORDER_REFLECT101|不含边界值倒序填充，如gfedcb\|abcdefgh\|gfedcba
>BORDER_TRANSPARENT|5|随机填充
>BORDER_ISOLATED|16|不关心区域外的部分

3. 三点变换函数

仿射变换又称为三点变换，如果知道变换前后两张图像中三个像素点坐标的对应关系，就可以求得仿射变换中的变换矩阵，OpenCV提供了利用三个对应像素点来确定矩阵的函数getAffineTransform()
```C++
Mat cv::getAffineTransform(const Point2f src[],const Point2f dst[])
```
该函数两个输入量都是存放浮点坐标的数组，在生成数组的时候像素点的输入顺序无关，但是需要保证像素点的对应关系，函数的返回值是一个2×3的变换矩阵。

```c++
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat img_1 = imread("picture.jpg");
	if(img_1.empty())
	{ 
		cout << "Fail to open!" << endl;
		return -1;
	}

	Mat img;
	resize(img_1, img, Size(img_1.cols / 2.0, img_1.rows / 2.0));
	Mat rotation_0, rotation_1, res0, res1;

	//旋转角，输出图像尺寸，旋转中心设置
	double angle = 30;
	Size dst_size(img.rows, img.cols);
	Point2f center(img.rows / 2.0, img.cols / 2.0);

	rotation_0 = getRotationMatrix2D(center, angle, 1);
	warpAffine(img, res0, rotation_0, dst_size);
	imshow("res0", res0);

	Point2f src_points[3];
	Point2f dst_points[3];
	src_points[0] = Point2f(0, 0);
	src_points[1] = Point2f(0, (float)(img.cols - 1));
	src_points[2] = Point2f((float)(img.rows - 1), (float)(img.cols - 1));

	dst_points[0] = Point2f((float)(img.rows * 0.11), (float)(img.cols * 0.20));
	dst_points[0] = Point2f((float)(img.rows * 0.15), (float)(img.cols * 0.70));
	dst_points[0] = Point2f((float)(img.rows * 0.81), (float)(img.cols * 0.85));
	rotation_1 = getAffineTransform(src_points, dst_points);
	warpAffine(img, res1, rotation_1, dst_size);
	imshow("res1", res1);

	waitKey();

	return 0;
}
```
## 2. 透视变换
透视变换是按照物体成像投影规律进行变换，即将物体重新投影到新的成像平面。
透视变换中，透视前的图像和透视后的图像之间的变换关系可以用一个3×3的变换矩阵表示，该矩阵可以通过两张图像中四个对应点的坐标求取，因此透视变换又称作“四点变换”。

1. 透视变换矩阵函数
```c++
Mat cv::getPerspectiveTransform (const Point2f src[],
                                 const Point2f dst[],
                                 int  solveMethod = DECOMP_LU
                                );
```
>src[]：原图像中的四个像素坐标。
>
>dst[]：目标图像中的四个像素坐标。
>
>solveMethod：选择计算透视变换矩阵方法的标志，

>solveMethod|值|含义
>-|-|-
>DECOMP_LU|0|最佳主轴元素的高斯消元法
>DECOMP_SVD|1|奇异值分解方法
>DECOMP_EIG|2|特征值分解法
>DECOMP_CHOLESKY|3|Cholesky分解法
>DECOMP_QR|4|QR分解法
>DECOMP_NORMOL|使用正规方程公式

2. 透视变换函数
```c++
void cv::warpPerspective(InputArray src,
                         OutputArray dst,
                         InputArray M,
                         Size dsize,
                         int  flags = INTER_LINEAR,
                         int  borderMode = BORDER_CONSTANT,
                         const Scalar & borderValue = Scalar()
                        );
```
>src：输入图像。
>
>dst：透视变换后输出图像，与src数据类型相同，但是尺寸与dsize相同。
>
>M：3×3的变换矩阵。
>
>dsize：输出图像的尺寸。
>
>flags：插值方法标志。
>
>borderMode：像素边界外推方法的标志。
>
>borderValue：填充边界使用的数值，默认情况下为0

```c++
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat img = imread("picture.jpg");
	if (img.empty())
	{
		cout << "Fail to open!" << endl;
		return -1;
	}

	Point2f src_point[4];
	Point2f dst_point[4];

	src_point[0] = Point2f(0.0, 1894.0);
	src_point[1] = Point2f(2173.0, 1966.0);
	src_point[2] = Point2f(0.0, 0.0);
	src_point[3] = Point2f(1802.0, 0.0);

	dst_point[0] = Point2f(0.0, 2262.0);
	dst_point[1] = Point2f(2364.0, 2262.0);
	dst_point[2] = Point2f(0.0, 0.0);
	dst_point[3] = Point2f(2364.0, 1.0);

	Mat matrix, res;
	matrix = getPerspectiveTransform(src_point, dst_point);
	warpPerspective(img, res, matrix, img.size());

	imshow("res", res);
	imshow("img", img);
	waitKey();
	return 0;
}
```

## 3. 极坐标变换
极坐标变换就是将图像在直角坐标系与极坐标系中互相变换
```c++
void cv::warpPolar(InputArray src,
                   OutputArray dst,
                   Size dsize,
                   Point2f center,
                   double  maxRadius,
                   int  flags
                  );
```
>src：原图像，可以是灰度图像或者彩色图像。
>
>dst：极坐标变换后输出图像，与原图像具有相同的数据类型和通道数。
>
>dsize：目标图像大小。
>
>center：极坐标变换时极坐标的原点坐标。
>
>maxRadius：变换时边界圆的半径，它也决定了逆变换时的比例参数。
>
>flags：插值方法与极坐标映射方法标志.

>flag|含义
>-|-
>WARP_POLAR_LINEAR|极坐标变换
>WARP_POLAR_LOG|半对数极坐标变换
>WARP_INVERSE_MAP|逆变换