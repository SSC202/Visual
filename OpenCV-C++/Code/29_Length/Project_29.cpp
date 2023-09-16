#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat img = imread("picture.jpeg", IMREAD_ANYDEPTH);
	if (img.empty())
	{
		cout << "Fail to open!" << endl;
		return -1;
	}
	Mat res1, res2, res3;
	threshold(img, img, 125, 255, THRESH_BINARY_INV);
	distanceTransform(img, res1, DIST_L1, 3, CV_32F);
	distanceTransform(img, res2, DIST_L2, 3);
	distanceTransform(img, res3, DIST_C, 3);

	imshow("res1", res1);
	imshow("res2", res2);
	imshow("res3", res3);

	waitKey(0);


	return 0;
}