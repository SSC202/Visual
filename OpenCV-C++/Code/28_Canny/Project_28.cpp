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

	Mat res;
	cvtColor(img, img, COLOR_BGR2GRAY);
	
	Canny(img, res, 200, 256);

	imshow("result", res);
	waitKey(0);

	return 0;
}