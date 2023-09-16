#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
	float a[12] = { 1,2,3,4,5,10,6,7,8,9,10,0 };

	Mat mat_1 = Mat(3, 4, CV_32FC1, a);
	Mat mat_2 = Mat(2, 3, CV_32FC2, a);

	double minVal, maxVal;
	Point minIdx, maxIdx;

	//单通道读取
	minMaxLoc(mat_1, &minVal, &maxVal, &minIdx, &maxIdx);
	cout << "最小值为：" << minVal << " 位置为：" << minIdx << endl;
	cout << "最大值为：" << maxVal << " 位置为：" << maxIdx << endl;

	//多通道读取
	Mat remat_2 = mat_2.reshape(1, 4);
	minMaxLoc(remat_2, &minVal, &maxVal, &minIdx, &maxIdx);
	cout << "最小值为：" << minVal << " 位置为：" << minIdx << endl;
	cout << "最大值为：" << maxVal << " 位置为：" << maxIdx << endl;

	Scalar Mean;

	Mean = mean(mat_1);
	cout << Mean[0] << endl;

	Mat M_Mean, stddev;
	meanStdDev(mat_2, M_Mean, stddev);

	cout << "Mean = " << M_Mean << "stddev = " << stddev << endl;
	system("pause");

	return 0;
}