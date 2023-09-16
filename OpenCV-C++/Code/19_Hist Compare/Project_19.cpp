#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
#define WIDTH 2
#define HIST_W 512
#define HIST_H 400
void drawHist(Mat& hist, Mat& hist_dst, int width, int hist_h, int hist_w, int type);

int main()
{
	Mat img_1 = imread("picture_1.jpg");
	Mat img_2 = imread("picture_2.jpg");

	if (img_1.empty() || img_2.empty())
	{
		cout << "Fail to open!" << endl;
		return -1;
	}
	cvtColor(img_1, img_1, COLOR_BGR2GRAY);
	cvtColor(img_2, img_2, COLOR_BGR2GRAY);

	Mat hist_1,hist_2,hist_dst_1,hist_dst_2;
	const int channel[1] = { 0 };
	float inrange[2] = { 0,255 };
	const float* range[1] = {inrange};
	const int dims[1] = { 256 };

	calcHist(&img_1, 1, channel, Mat(), hist_1, 1, dims, range);
	calcHist(&img_2, 1, channel, Mat(), hist_2, 1, dims, range);

	drawHist(hist_1, hist_dst_1, WIDTH, HIST_H, HIST_W, NORM_L1);
	drawHist(hist_2, hist_dst_2, WIDTH, HIST_H, HIST_W, NORM_L1);
	system("cls");
	double alpha = compareHist(hist_dst_1, hist_dst_2, HISTCMP_CORREL);
	cout << "alpha = " << alpha << endl;

//	imshow("img_1", img_1);
//	imshow("img_2", img_2);
	waitKey();

	return 0;
}

void drawHist(Mat &hist,Mat &hist_dst,int width,int hist_h,int hist_w,int type)
{
	Mat Img_image = Mat::zeros(hist_h, hist_w, CV_8UC3);
	normalize(hist, hist_dst, 1, 0, type, -1, Mat());
	for (int i = 1; i <= hist.rows; ++i)
	{
		rectangle(Img_image, Point(width * (i - 1), hist_h - 1), Point(width * i - 1, hist_h - cvRound(hist_h * hist_dst.at<float>(i - 1)) - 1), Scalar(255, 255, 255), -1);
	}
}