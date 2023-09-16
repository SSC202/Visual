#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#define LUT_MAX 256

int main()
{
	Mat img = imread("picture.jpg");
	if (img.empty())
	{
		cout << "Fail to open!" << endl;
		return -1;
	}

	uchar LUT_C1[LUT_MAX];
	for (int i = 0; i < 256; ++i) 
	{
		if (i <= 100)
			LUT_C1[i] = 0;
		else if (i > 100 && i <= 200)
			LUT_C1[i] = 100;
		else 
			LUT_C1[i] = 255;
	}
	Mat lut_c1(1, LUT_MAX, CV_8UC1, LUT_C1);

	uchar LUT_C2[LUT_MAX];
	for (int i = 0; i < 256; ++i)
	{
		if (i <= 100)
			LUT_C2[i] = 0;
		else if (i > 100 && i <= 200)
			LUT_C2[i] = 100;
		else
			LUT_C2[i] = 255;
	}
	Mat lut_c2(1, LUT_MAX, CV_8UC1, LUT_C2);

	uchar LUT_C3[LUT_MAX];
	for (int i = 0; i < 256; ++i)
	{
		if (i <= 100)
			LUT_C3[i] = 0;
		else if (i > 100 && i <= 200)
			LUT_C3[i] = 100;
		else
			LUT_C3[i] = 255;
	}
	Mat lut_c3(1, LUT_MAX, CV_8UC1, LUT_C3);

	Mat LUT_List;
	vector<Mat> mergeMats;
	mergeMats.push_back(lut_c1);
	mergeMats.push_back(lut_c2);
	mergeMats.push_back(lut_c3);
	merge(mergeMats, LUT_List);

	Mat res1, res2;
	LUT(img, lut_c1, res1);
	LUT(img, LUT_List, res2);

	imshow("res1", res1);
	imshow("res2", res2);
	waitKey();

	return 0;
}
