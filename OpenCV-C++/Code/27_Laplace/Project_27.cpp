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

	Mat dst, result;

	cvtColor(img, img, COLOR_BGR2GRAY);
	GaussianBlur(img, dst, Size(3, 3), 10, 20);
	Laplacian(dst, result, -1, 3);
	convertScaleAbs(result, result);

	imshow("result", result);
	waitKey(0);


	return 0;
}