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

	Mat kernel_1 = (Mat_<float>(1, 2) << 1, -1);
	Mat kernel_2 = (Mat_<float>(1, 3) << 1, 0, -1);
	Mat kernel_3 = (Mat_<float>(3, 1) << 1, 0, -1);
	Mat kernelXY = (Mat_<float>(2, 2) << 1, 0, 0, -1);  
	Mat kernelYX = (Mat_<float>(2, 2) << 0, -1, 1, 0);  

	Mat dst;

	cvtColor(img, dst, COLOR_BGR2GRAY);

	Mat result1, result2, result3, result4, result5, result6;

	filter2D(img, result1, CV_16S, kernel_1);
	convertScaleAbs(result1, result1);

	filter2D(img, result2, CV_16S, kernel_2);
	convertScaleAbs(result2, result2);

	filter2D(img, result3, CV_16S, kernel_3);
	convertScaleAbs(result3, result3);

	result6 = result2 + result3;

	filter2D(img, result4, CV_16S, kernelXY);
	convertScaleAbs(result4, result4);


	filter2D(img, result5, CV_16S, kernelYX);
	convertScaleAbs(result5, result5);


	imshow("result1", result1);
	imshow("result2", result2);
	imshow("result3", result3);
	imshow("result4", result4);
	imshow("result5", result5);
	imshow("result6", result6);
	waitKey(0);

	return 0;
}