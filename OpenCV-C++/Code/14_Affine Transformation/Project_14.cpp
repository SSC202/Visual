#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat img_1 = imread("picture.jpg");
	if(img_1.empty())
	{ 
		cout << "Fail to open!" << endl;
		return -1;
	}

	Mat img;
	resize(img_1, img, Size(img_1.cols / 2.0, img_1.rows / 2.0));
	Mat rotation_0, rotation_1, res0, res1;

	//旋转角，输出图像尺寸，旋转中心设置
	double angle = 30;
	Size dst_size(img.rows, img.cols);
	Point2f center(img.rows / 2.0, img.cols / 2.0);

	rotation_0 = getRotationMatrix2D(center, angle, 1);
	warpAffine(img, res0, rotation_0, dst_size);
	imshow("res0", res0);

	Point2f src_points[3];
	Point2f dst_points[3];
	src_points[0] = Point2f(0, 0);
	src_points[1] = Point2f(0, (float)(img.cols - 1));
	src_points[2] = Point2f((float)(img.rows - 1), (float)(img.cols - 1));

	dst_points[0] = Point2f((float)(img.rows * 0.11), (float)(img.cols * 0.20));
	dst_points[0] = Point2f((float)(img.rows * 0.15), (float)(img.cols * 0.70));
	dst_points[0] = Point2f((float)(img.rows * 0.81), (float)(img.cols * 0.85));
	rotation_1 = getAffineTransform(src_points, dst_points);
	warpAffine(img, res1, rotation_1, dst_size);
	imshow("res1", res1);

	waitKey();

	return 0;
}