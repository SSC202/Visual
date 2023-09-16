#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

void AddSalt(Mat& img, int n);

int main()
{
	Mat img = imread("picture.jpg");
	if (img.empty())
	{
		cout << "Fail to open!" << endl;
		return -1;
	}

	Mat gray;


	cvtColor(img, gray, COLOR_BGR2GRAY);

	imshow("img", img);
	imshow("gray", gray);

	AddSalt(img, 10000);
	AddSalt(gray, 10000);

	imshow("img", img);
	imshow("gray", gray);

	waitKey();



	return 0;
}

void AddSalt(Mat& img, int n)
{
	for (int i = 0; i < n; i++)
	{
		int a = cvflann::rand_int(img.rows, 0);
		int b = cvflann::rand_int(img.cols, 0);

		int flag = cvflann::rand_int(32767, 0) % 2;
		if (flag == 1)
		{
			if (img.type() == CV_8UC1)
			{
				img.at<uchar>(a, b) = 255;
			}
			else if (img.type() == CV_8UC3)
			{
				img.at<Vec3b>(a, b)[0] = 255;
				img.at<Vec3b>(a, b)[1] = 255;
				img.at<Vec3b>(a, b)[2] = 255;
			}
		}
		if (flag == 0)
		{
			if (img.type() == CV_8UC1)
			{
				img.at<uchar>(a, b) = 0;
			}
			else if (img.type() == CV_8UC3)
			{
				img.at<Vec3b>(a, b)[0] = 0;
				img.at<Vec3b>(a, b)[1] = 0;
				img.at<Vec3b>(a, b)[2] = 0;
			}
		}
	}
}