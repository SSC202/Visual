# OpenCV 13_图像修复

在实际应用或者工程中，图像常常会收到噪声的干扰，例如在拍照时镜头上存在灰尘或者飞行的小动物，这些干扰会导致拍摄到的图像出现部分内容被遮挡的情况。对于较为久远的图像，可能只有实体图像而没有数字存储形式的底板，因此相片在保存和运输过程中可能产生划痕，导致图像中信息的损坏和丢失。

```c++
void cv::inpaint(InputArray  src,
                 InputArray  inpaintMask,
                 OutputArray  dst,
                 double  inpaintRadius,
                 int  flags 
                );
```

>src：输入待修复图像，当图像为单通道时，数据类型可以是CV_8U、CV_16U或者CV_32F，当图像为三通道时数据类型必须是CV_8U。
>
>inpaintMask：修复掩模，数据类型为CV_8U的单通道图像，与待修复图像具有相同的尺寸。
>
>dst：修复后输出图像，与输入图像具有相同的大小和数据类型。
>
>inpaintRadius：算法考虑的每个像素点的圆形邻域半径。
>
>flags：修复方法标志

该函数虽然可以对图像受污染区域进行修复，但是需要借助污染边缘区域的像素信息，离边缘区域越远的像素估计出的准确性越低，因此如果受污染区域较大，修复的效果就会降低。

```c++
#include <opencv2\opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat img = imread("picture.jpg");
	Mat img_mask;
	if (img.empty())
	{
		cout << "请确认图像文件名称是否正确" << endl;
		return -1;
	}
	imshow("img", img);


	//转换为灰度图
	Mat imgGray;
	cvtColor(img, imgGray, COLOR_RGB2GRAY, 0);


	//通过阈值处理生成Mask掩模
	threshold(imgGray, img_mask, 200, 255, THRESH_BINARY);


	//对Mask膨胀处理，增加Mask面积
	Mat Kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(img_mask, img_mask, Kernel);

	//图像修复
	Mat imgInpaint;
	inpaint(img, img_mask, imgInpaint, 5, INPAINT_NS);


	//显示处理结果
	imshow("imgMask", img_mask);
	imshow("imgInpaint", imgInpaint);
	waitKey();
	return 0;
}
```
