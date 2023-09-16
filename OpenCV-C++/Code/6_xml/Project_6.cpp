#include<iostream>
#include<vector>
#include<string>
//#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
	cout << "OpenCv Version: " << CV_VERSION << endl;
	FileStorage file;
	string filename = "text.yaml";

	file.open(filename, FileStorage::WRITE);

	if (!file.isOpened())
	{
		cout << "Fail to open!" << endl;
		return -1;
	}

	Mat mat = Mat::eye(3, 3, CV_8U);

	file.write("Mat", mat);

	float x = 100;
	file << "x" << x;

	string str = "Test";
	file << "str" << str;

	file <<  "Array" << "[" << 1 << 2 << 3 << 4 << "]";

	file << "Time" << "{" << "year" << 2022 << "month" << 2 << "day" << 17 << "d_time" << "[" << 0 << 1 << 2 << 3 << "]" << "}";

	file.release();

	FileStorage rfile;
	rfile.open(filename, FileStorage::READ);

	if (!rfile.isOpened())
	{
		cout << "Fail to open!" << endl;
		return -1;
	}

	float x_r;
	rfile["x"] >> x_r;
	cout << "x_r = " << x_r << endl;

	string str_r;
	rfile["str"] >> str_r;
	cout << "str_r = " << str_r << endl;

	FileNode node = rfile["Array"];
	cout << "Array = [";
	for (FileNodeIterator it = node.begin(); it != node.end(); it++)
	{
		int i;
		*it >> i;
		cout << i << " ";
	}
	cout << "]" << endl;

	Mat mat_r;
	rfile["Mat"] >> mat_r;
	cout << "mat_r = " << mat_r << endl;

	FileNode t_node = rfile["Time"];
	int year_r = (int)t_node["year"];
	int month_r = (int)t_node["month"];
	int day_r = (int)t_node["day"];
	int hour_r = (int)t_node["d_time"][0];
	int minute_r = (int)t_node["d_time"][1];
	int second_r = (int)t_node["d_time"][2];

	cout << "Time: " << year_r << "." << month_r << "." << day_r << "." << hour_r << ":" << minute_r << ":" << second_r << endl;


	rfile.release();
	system("pause");
	return 0;
}