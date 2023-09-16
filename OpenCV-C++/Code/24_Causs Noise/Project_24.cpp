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

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	imshow("img", img);
	imshow("gray", gray);

	Mat noise = Mat::zeros(Size(img.cols, img.rows), CV_8UC3);
    Mat g_noise = Mat::zeros(Size(img.cols, img.rows), CV_8UC1);

	RNG rng;
	rng.fill(noise, RNG::NORMAL, 10, 20);
	rng.fill(g_noise, RNG::NORMAL, 15, 30);

	imshow("img_noise", noise);
	imshow("gray_noise", g_noise);

	img = img + noise;
	gray = gray + g_noise;

	imshow("img", img);
	imshow("gray", gray);

	waitKey();
	return 0;
}