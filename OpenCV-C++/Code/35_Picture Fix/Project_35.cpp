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
