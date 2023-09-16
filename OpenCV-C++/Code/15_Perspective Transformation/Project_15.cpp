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

	Point2f src_point[4];
	Point2f dst_point[4];

	src_point[0] = Point2f(0.0, 1894.0);
	src_point[1] = Point2f(2173.0, 1966.0);
	src_point[2] = Point2f(0.0, 0.0);
	src_point[3] = Point2f(1802.0, 0.0);

	dst_point[0] = Point2f(0.0, 2262.0);
	dst_point[1] = Point2f(2364.0, 2262.0);
	dst_point[2] = Point2f(0.0, 0.0);
	dst_point[3] = Point2f(2364.0, 1.0);

	Mat matrix, res;
	matrix = getPerspectiveTransform(src_point, dst_point);
	warpPerspective(img, res, matrix, img.size());

	imshow("res", res);
	imshow("img", img);
	waitKey();
	return 0;
}