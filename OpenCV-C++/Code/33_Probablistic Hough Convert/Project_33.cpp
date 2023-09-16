#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<vector>
using namespace std;
using namespace cv;

void Read_Img(Mat& img, string str)
{
    img = imread(str);
    if (img.empty())
    {
        cout << "Fail to open :" << str << " " << endl;
        exit(-1);
    }
}

int main()
{
    Mat img, edge;
    Mat imgl;
    Read_Img(img, "picture.jpg");
    img.copyTo(imgl);
    cvtColor(img, img, COLOR_BGR2GRAY);
    Canny(img, edge, 50, 200);
    threshold(edge, edge, 170, 255, THRESH_BINARY);

    vector<Vec4i>lines;
    HoughLinesP(edge, lines, 1., CV_PI / 100, 50, 10);

    for (int i = 0; i < lines.size(); i++)
    {
        Vec4i point_ = lines[i];
        line(imgl, Point(point_[0], point_[1]), Point(point_[2], point_[3]), Scalar(0, 255, 0));
    }


    imshow("edge", edge);
    imshow("img", img);
    imshow("img1", imgl);
    waitKey(0);
    return 0;

}

