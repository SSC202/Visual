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
	//直方图数据设置
	Mat hist;
	const int channels[1] = { 0 };           //通道索引
	float inRanges[2] = { 0,255 };
	const float* ranges[1] = { inRanges };   //像素灰度值范围
	const int dims[1] = { 256 };             //直方图维度（像素灰度值最大值）
	//直方图生成
	calcHist(&img, 1, channels, Mat(), hist, 1, dims, ranges);

	//直方图绘制
	int hist_w = 512;
	int hist_h = 400;
	int width = 2;
	Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
	for (int i = 1; i <= hist.rows; ++i)
	{
		rectangle(histImage, Point(width * (i - 1), hist_h - 1), Point(width * i - 1, hist_h - cvRound(hist.at<float>(i - 1) / 20)), Scalar(255, 255, 255), -1);
	}
	namedWindow("histImage", WINDOW_AUTOSIZE);
	imshow("histImage", histImage);
	imshow("gray", gray);
	waitKey(0);
	return 0;
}