#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

int main()
{

	Mat M(2, 2, CV_8UC3, Scalar(0, 0, 255));
	cout << "M = " << endl;
	cout << M << endl;

	Mat A = Mat::ones(3, 3, CV_32FC1);
	Mat B = Mat::ones(3, 3, CV_32FC1);
	Mat AB;

	// 使用at进行赋值
	for (int i = 0; i < A.rows; i++)
	{
		for (int j = 0; j < A.cols; j++)
		{
			A.at<float>(i, j) = (float)(i + j);
		}
	}

	for (int i = 0; i < B.rows; i++)
	{
		for (int j = 0; j < B.cols; j++)
		{
			B.at<float>(i, j) = (float)(i + j);
		}
	}

	// 矩阵乘法
	AB = A * B;

	cout <<"AB = " << AB << endl;
	cout <<"A = " << A << endl;
	cout <<"B = " << B << endl;

	// 矩阵点乘
	double A_dot_B;

	A_dot_B = A.dot(B);
	cout << "A_dot_B = " << A_dot_B << endl;

	// 矩阵对应位乘法
	Mat A_mul_B;

	A_mul_B = A.mul(B);

	cout << "A_mul_B = " << A_mul_B << endl;

	system("pause");
	return 0;
}