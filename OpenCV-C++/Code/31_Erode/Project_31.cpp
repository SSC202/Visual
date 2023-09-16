#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

void Read_Img(Mat& img, string str)
{
	img = imread(str);
	if (img.empty())
	{
		cout << "Fail to open :" << str << " " << endl;
		exit(-1);
	}
}

void drawstate(Mat& img, int number, Mat center, Mat stats, string str)
{
	RNG rng(10086);
	vector<Vec3b> color;
	for (int i = 0; i < number; i++)
	{
		Vec3b vec3 = Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		color.push_back(vec3);
	}

	for (int i = 1; i < number; i++)
	{
		int center_x = center.at<double>(i, 0);
		int center_y = center.at<double>(i, 1);

		int x = stats.at<int>(i, CC_STAT_LEFT);
		int y = stats.at<int>(i, CC_STAT_TOP);
		int h = stats.at<int>(i, CC_STAT_HEIGHT);
		int w = stats.at<int>(i, CC_STAT_WIDTH);

		circle(img, Point(center_x, center_y), 2, Scalar(0, 0, 255), 1, 0, 0);

		Rect rect(x, y, w, h);
		rectangle(img, rect, color[i], 1, 0, 0);
		putText(img, format("%d", i), Point(center_x, center_y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);

	}
	imshow(str, img);
}

int main()
{
	Mat img, imgc, res, stats, center;
	Read_Img(img, "picture.jpeg");
	copyTo(img, imgc, img);

	cvtColor(img, img, COLOR_BGR2GRAY);
	threshold(img, img, 50, 255, THRESH_BINARY);

	Mat struct_1 = getStructuringElement(0, Size(3, 3));
	erode(img, img, struct_1);
	int number = connectedComponentsWithStats(img, res, stats, center, 8, CV_16U);
	drawstate(img, number, center, stats, "res");

	imshow("imgc", imgc);



	waitKey(0);







	return 0;
}