#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<vector>

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
	Mat dst;
	resize(img, dst, Size(img.cols*2, img.rows*2));

	vector<Mat> Gauss, Lap;
	Gauss.push_back(dst);
	int level = 2;

	//高斯金字塔构造
	for (int i = 0; i < level; ++i)
	{
		Mat gauss;
		pyrDown(Gauss[i], gauss);
		Gauss.push_back(gauss);
	}

	//拉普拉斯金字塔构造
	for (int i = Gauss.size()-1 ; i > 0; --i)
	{
		Mat lap, upgauss;
		if (i == Gauss.size()-1)
		{
			Mat down;
			pyrDown(Gauss[i], down);
			pyrUp(down, upgauss);
			resize(upgauss, upgauss, Size(Gauss[i].cols, Gauss[i].rows));
			lap = Gauss[i] - upgauss;
			Lap.push_back(lap);
		}
		pyrUp(Gauss[i], upgauss);
		resize(upgauss, upgauss, Size(Gauss[i - 1].cols, Gauss[i - 1].rows));
		lap = Gauss[i - 1] - upgauss;
		Lap.push_back(lap);
	}

	for (int i = 0; i < Gauss.size(); ++i) {
		string name = to_string(i);
		imshow("G" + name, Gauss[i]);
		imshow("L" + name, Lap[i]);
	}
	waitKey();
	
	return 0;
}


