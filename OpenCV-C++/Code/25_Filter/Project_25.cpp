#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

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

	AddSalt(img, 10000);

	Mat res1,res2,res3,res4;

	blur(img, res1, Size(9, 9));
	boxFilter(img, res2,-1,Size(9,9));
	sqrBoxFilter(img, res3, -1, Size(3, 3), Point(-1, -1), false, BORDER_CONSTANT);
	GaussianBlur(img, res4, Size(5, 5), 10, 20);

	imshow("img", img);
	imshow("res1", res1);
	imshow("res2", res2);
	imshow("res3", res3);
	imshow("res4", res4);
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