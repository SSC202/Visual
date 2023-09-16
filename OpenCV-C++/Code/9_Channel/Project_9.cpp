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

	Mat HSV, dst;
	resize(img, dst, Size(img.cols * 0.5, img.rows * 0.5));

	cvtColor(dst, HSV, COLOR_BGR2HSV);

	Mat img0, img1, img2;
	Mat vimg0, vimg1, vimg2;
	Mat res1, res2, res3;

	Mat imgs[3];
	split(dst, imgs);
	img0 = imgs[0];
	img1 = imgs[1];
	img2 = imgs[2];

	imshow("BGR-R", img0);
	imshow("BGR-G", img1);
	imshow("BGR-B", img2);
	
	//res1为5通道图片，无法调用imshow()函数（导致数据溢出）
	imgs[2] = dst;
	merge(imgs, 3, res1);

	Mat zeros = Mat::zeros(dst.rows, dst.cols, CV_8UC1);
	imgs[0] = zeros;
	imgs[2] = zeros;
	merge(imgs, 3, res2);
	
	imshow("Result_1", res2);

	vector<Mat> hsv;
	split(HSV, hsv);
	vimg0 = hsv.at(0);
	vimg1 = hsv.at(1);
	vimg2 = hsv.at(2);
	imshow("HSV-H", vimg0);
	imshow("HSV-S", vimg1);
	imshow("HSV-V", vimg2);

	hsv.push_back(HSV);
	merge(hsv, res3);
	//res3为5通道图片，无法调用imshow()函数（导致数据溢出）

	waitKey();



	return 0;
}