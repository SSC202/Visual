#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void CreateMat(Mat& mat)
{
	CV_Assert(mat.channels() == 4);
	for (int i = 0; i < mat.rows; ++i)
	{
		for (int j = 0; j < mat.cols; ++j)
		{
			Vec4b& channel = mat.at<Vec4b>(i, j);
			channel[0] = UCHAR_MAX;
			channel[1] = saturate_cast<uchar>((float(mat.cols - j)) / ((float)mat.cols) * UCHAR_MAX);
			channel[2] = saturate_cast<uchar>((float(mat.rows - i)) / ((float)mat.rows) * UCHAR_MAX);
			channel[3] = saturate_cast<uchar>(0.5 * (channel[1] + channel[2]));
		}
	}
}

int main()
{
	cout << "OpenCV Version: " << CV_VERSION << endl;
	Mat mat(480, 640, CV_8UC4, Scalar(0, 0, 0, 0));

	CreateMat(mat);

	vector<int>comprocession_params;
	comprocession_params.push_back(IMWRITE_PNG_COMPRESSION);
	comprocession_params.push_back(9);

	bool result = imwrite("picture.png", mat, comprocession_params);

	if (!result)
	{
		cout << "failed!" << endl;
		return -1;
	}
	cout << "Success!" << endl;
	imshow("picture.png", mat);
	waitKey(0);
	return 0;
}