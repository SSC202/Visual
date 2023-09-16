#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#define WIDTH 2
#define HIST_W 512
#define HIST_H 400
void createHist(Mat& img, Mat& hist);
void drawHist(Mat& hist, Mat& hist_dst, int type,string str, int width = WIDTH, int hist_h = HIST_H, int hist_w = HIST_W);

int main()
{
	Mat img = imread("picture.jpg");
	if (img.empty())
	{
		cout << "Fail to open!" << endl;
		return -1;
	}

	cvtColor(img, img, COLOR_BGR2GRAY);
	Mat hist,hist_dst,eimg,ehist,ehist_dst;
	equalizeHist(img, eimg);

	createHist(img, hist);
	drawHist(hist, hist_dst, NORM_L1,"hist");
	createHist(eimg, ehist);
	drawHist(ehist, ehist_dst, NORM_L1,"ehist");

	double alpha = compareHist(ehist_dst, hist_dst, HISTCMP_CORREL);
	system("cls");
	cout << "alpha = " << alpha << endl;
	
	imshow("eimg", eimg);

	waitKey();

	return 0;
}

void createHist(Mat& img,Mat &hist)
{
	const int channel[1] = { 0 };
	float inrange[2] = { 0,255 };
	const float* range[1] = { inrange };
	const int dims[1] = { 256 };
	calcHist(&img, 1, channel, Mat(), hist, 1, dims, range);
}

void drawHist(Mat& hist, Mat& hist_dst, int type,string str, int width, int hist_h, int hist_w)
{
	Mat Img_image = Mat::zeros(hist_h, hist_w, CV_8UC3);
	normalize(hist, hist_dst, 1, 0, type, -1, Mat());
	for (int i = 1; i <= hist.rows; ++i)
	{
		rectangle(Img_image, Point(width * (i - 1), hist_h - 1), Point(width * i - 1, hist_h - cvRound(hist_h * hist_dst.at<float>(i - 1)) - 1), Scalar(255, 255, 255), -1);
	}
	imshow(str, Img_image);
}