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


