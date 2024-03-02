#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

bool draw_flag = false;
bool mode = false;

Mat img;

void tracebar_callback(int value, void *userdata)
{
    ;
}

void mouse_callback(int event, int x, int y, int flags, void *param)
{
    int r, g, b;
    r = getTrackbarPos("R", "img");
    g = getTrackbarPos("G", "img");
    b = getTrackbarPos("B", "img");
    if (event == EVENT_LBUTTONDOWN)
    {
        draw_flag = true;
    }
    else if ((event == EVENT_MOUSEMOVE) && (flags == EVENT_LBUTTONDOWN))
    {
        if (draw_flag == true)
        {
            if (mode == true)
            {
                circle(img, Point(x, y), 10, Scalar(b, g, r), -1, LINE_AA);
            }
            else
            {
                rectangle(img, Point(x - 5, y - 5), Point(x + 5, y + 5), Scalar(b, g, r), -1, LINE_AA);
            }
        }
    }
    else if (event == EVENT_LBUTTONUP)
    {
        draw_flag = false;
    }
}

int main()
{
    namedWindow("img");
    img = Mat::zeros(Size(640, 480), CV_8UC3);
    createTrackbar("R", "img", 0, 255, tracebar_callback);
    createTrackbar("G", "img", 0, 255, tracebar_callback);
    createTrackbar("B", "img", 0, 255, tracebar_callback);
    setMouseCallback("img", mouse_callback);
    for (;;)
    {
        imshow("img", img);
        int key = waitKey(1);
        if (key == 27)
        {
            break;
        }
    }
}