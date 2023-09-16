#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat img = imread("picture.jpg");
	Mat plate = imread("plate.jpg");

	if (img.empty() || plate.empty())
	{
		cout << "Fail to open!" << endl;
		return -1;
	}

	Mat res;
	matchTemplate(img, plate, res, TM_CCOEFF_NORMED);
	double minVal, maxVal;
	Point min, max;

	minMaxLoc(res, &minVal, &maxVal, &min, &max);
	rectangle(img, Rect(max.x, max.y, plate.cols, plate.rows), Scalar(0, 0, 255), 2);
	imshow("img", img);
	imshow("plate", plate);
	imshow("result", res);
	waitKey();



	return 0;
}