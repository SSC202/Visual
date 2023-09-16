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

	Mat src;
	cvtColor(img, src, COLOR_BGR2GRAY);

	//直方图数据配置
	Mat hist;
	const int channels[1] = { 0 };
	float inranges[2] = { 0,255 };
	const float* ranges[1] = { inranges };
	const int dims[1] = {256};

	calcHist(&src, 1, channels, Mat(), hist, 1, dims, ranges);

	int hist_w = 512;
	int hist_h = 400;
	int width = 2;

	Mat Img_image_1 = Mat::zeros(hist_h, hist_w, CV_8UC3);
	Mat Img_image_2 = Mat::zeros(hist_h, hist_w, CV_8UC3);
	Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
	Mat hist_1, hist_2;
	normalize(hist, hist_1, 1, 0, NORM_L2, -1, Mat());
	normalize(hist, hist_2, 1, 0, NORM_L1, -1, Mat());

	for (int i = 1; i <= hist.rows; ++i)
	{
		rectangle(Img_image_1, Point(width * (i - 1), hist_h - 1),Point(width * i - 1, hist_h - cvRound(hist_h * hist_1.at<float>(i - 1)) - 1),Scalar(255, 255, 255), -1);
	}

	for (int i = 1; i <= hist.rows; ++i)
	{
		rectangle(Img_image_2, Point(width * (i - 1), hist_h - 1), Point(width * i - 1, hist_h - cvRound(hist_h * hist_2.at<float>(i - 1)) - 1), Scalar(255, 255, 255), -1);
	}

	for (int i = 1; i <= hist.rows; ++i)
	{
		rectangle(histImage, Point(width * (i - 1), hist_h - 1), Point(width * i - 1, hist_h - cvRound(hist_h * hist.at<float>(i - 1)) - 1), Scalar(255, 255, 255), -1);
	}
	imshow("IMG_HIST", histImage);
	imshow("histImage_L2", Img_image_1);
	imshow("histImage_L1", Img_image_2);
	imshow("img", src);
	waitKey();

	return 0;
}