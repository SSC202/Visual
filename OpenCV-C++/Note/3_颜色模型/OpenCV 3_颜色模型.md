# OpenCV 3_颜色模型

## 1. 颜色模型简介

1. RGB 模型

B ---- 蓝色（第一通道）
G ---- 绿色（第二通道）
R ---- 红色（第三通道）

三个通道对于颜色描述的范围是相同的，因此RGB颜色模型的空间构成是一个立方体。

在RGB颜色模型中，所有的颜色都是由这三种颜色通过不同比例的混合得到，如果三种颜色分量都为0，则表示为黑色，如果三种颜色的分量相同且都为最大值，则表示为白色。

每个通道都表示某一种颜色由0到1的过程，不同位数的图像表示将这个颜色变化过程细分成不同的层级，例如8U3C格式的图像每个通道将这个过程量化成256个等级，分别由0到255表示。

>在这个模型的基础上增加第四个通道即为RGBA模型，第四个通道表示颜色的透明度，当没有透明度需求的时候，RGBA模型就会退化成RGB模型。

2. YUV 模型

YUV模型是电视信号系统所采用的颜色编码方式。这三个变量分别表示是像素的亮度（Y）以及红色分量与亮度的信号差值（U）和蓝色与亮度的差值（V）。

$$
\begin{cases}
\ Y=0.299R+0.587G+0.114B \\
\ U=-0.147R-0.289G+0.436B \\
\ V =0.615R-0.515G-0.100B
\end{cases}
$$

3. HSV模型

HSV是色度（Hue）、饱和度（Saturation）和亮度（Value）的简写，该模型通过这三个特性对颜色进行描述。
>色度是色彩的基本属性。
>
>饱和度是指颜色的纯度，饱和度越高色彩越纯越艳，饱和度越低色彩则逐渐地变灰变暗，饱和度的取值范围是由0到100%；
>
>亮度是颜色的明亮程度，其取值范围由0到计算机中允许的最大值。

![NULL](picture_1.jpg)

4. Lab模型

Lab颜色模型弥补了RGB模型的不足，是一种设备无关的颜色模型，是一种基于生理特征的颜色模型。在模型中L表示亮度（Luminosity），a和b是两个颜色通道，两者的取值区间都是由-128到+127，其中a通道数值由小到大对应的颜色是从绿色变成红色，b通道数值由小到大对应的颜色是由蓝色变成黄色。

5. GRAY模型

GRAY模型并不是一个彩色模型，他是一个灰度图像的模型，其命名使用的是英文单词gray的全字母大写。灰度图像只有单通道，灰度值根据图像位数不同由0到最大依次表示由黑到白。
$$
Gray = R*0.3+G*0.59+B*0.11
$$

## 2. 颜色模型转换函数
```c++
void cv::cvtColor(InputArray src,OutputArray dst,int code,int dstCn = 0);
```
>src：待转换颜色模型的原始图像。
>
>dst：转换颜色模型后的目标图像。
>
>code：颜色空间转换的标志，如由RGB空间到HSV空间。
>
>dstCn：目标图像中的通道数，如果参数为0，则从src和代码中自动导出通道数。

>code|值|作用
>-|-|-
>COLOR_BGR2BGRA|0|对RGB图像添加Alpha通道
>COLOR_BGR2RGB|4|彩色通道颜色顺序更改
>COLOR_BGR2GRAY|10|彩色图像转为灰色图像
>COLOR_GRAY2BGR|8|灰度图像转换为彩色图像
>COLOR_BGR2YUV|82|BGR模型转为YUV模型
>COLOR_YUV2BGR|84|YUV模型转换为BGR模型
>COLOR_BGR2HSV|40|RGB模型转换为HSV模型
>COLOR_HSV2BGR|54|HSV模型转换为BGR模型
>COLOR_BGR2Lab|44|RGB模型转换为Lab模型
>COLOR_Lab2BGR|56|Lab模型转换为RGB模型
```c++
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int main()
{
	Mat img = imread("picture.jpg");
	if (img.empty())
	{
		cout << "Fail to open!" << endl;
		return -1;
	}

	Mat dst;
	resize(img, dst, Size(img.cols * 0.5, img.rows * 0.5));

	Mat Gray, HSV, YUV, Lab, Img_32;

	dst.convertTo(Img_32, CV_32F, 1.0 / 255);

	cvtColor(Img_32, Gray, COLOR_RGB2GRAY);
	cvtColor(Img_32, HSV, COLOR_RGB2HSV);
	cvtColor(Img_32, YUV, COLOR_RGB2YUV);
	cvtColor(Img_32, Lab, COLOR_RGB2Lab);

	imshow("img", dst);
	imshow("Gray", Gray);
	imshow("HSV", HSV);
	imshow("YUV", YUV);
	imshow("Lab", Lab);
	waitKey();

	return 0;


}
```