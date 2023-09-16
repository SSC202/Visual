#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main()
{
    system("color F0");                                                 //更改输出界面颜色
    Mat img = imread("picture.jpeg");
    if (img.empty()) {
        cout << "Fail to open!" << endl;
        return -1;
    }
    imshow("原图", img);
    Mat gray, binary;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(3, 3), 2, 2);                         //平滑滤波
    threshold(gray, binary, 135, 255, THRESH_BINARY_INV);                   //自适应二值化

    //轮廓发现与绘制
    vector<vector<Point>>contours;                                      //轮廓
    vector<Vec4i>hierachy;                                              //存放轮廓结构变量
    findContours(binary, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    //绘制轮廓
    for (int i = 0; i < contours.size(); ++i) {
        drawContours(img, contours, i, Scalar(0, 0, 255), 2, 8);
    }

    //输出轮廓结构描述
    for (int i = 0; i < hierachy.size(); ++i) {
        cout << hierachy[i] << endl;
    }
    //显示结果
    imshow("res", img);
    waitKey(0);
    return 0;
}
