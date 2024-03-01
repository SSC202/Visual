#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
    Mat img = imread("test.jpg");
    imshow("res", img);
    waitKey();
    return 0;
}
