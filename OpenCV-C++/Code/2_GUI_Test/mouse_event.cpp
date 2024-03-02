#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat img = Mat::zeros(Size(640, 480), CV_8UC3);
bool draw_flag, mode;
int ix, iy;

/**
 * @brief   鼠标事件回调函数
 */
void mouse_event_callback(int event, int x, int y, int flags, void *param)
{

    switch (event)
    {
    case EVENT_LBUTTONDOWN:
        draw_flag = true;
        ix = x;
        iy = y;
        break;
    case EVENT_MOUSEMOVE:
        if (flags == EVENT_FLAG_LBUTTON)
        {
            if (mode == true)
            {
                circle(img, Point(x, y), 10, Scalar(0, 255, 0), -1, LINE_AA);
            }
            else if (mode == false)
            {
                rectangle(img, Point(x - 10, y - 10), Point(x + 10, y + 10), Scalar(255, 0, 0), -1, LINE_AA);
            }
        }
        break;
    case EVENT_LBUTTONUP:
        draw_flag = false;
    default:
        break;
    }
}

int main()
{
    namedWindow("img");
    setMouseCallback("img", mouse_event_callback);
    for (;;)
    {
        imshow("img", img);
        int key = waitKey(1);
        if (key == 27)
        {
            break;
        }
    }
    destroyAllWindows();
    return 0;
}