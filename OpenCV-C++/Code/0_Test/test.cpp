/*从相机中读取视频*/
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
 
using namespace cv;
using namespace std;
 
int main()
{
    //打开捕获器（0-系统默认摄像头）
    VideoCapture cap(1); 
    Mat frame;
    //打开失败
    if (!cap.isOpened()) {
        cerr << "Cannot open camera";
        return -1;
    }
    //打开成功
    while (true) {
        //读取视频帧
        cap.read(frame);
        //显示图像
        imshow("Pig", frame);
        //监听键盘，按任意键退出
        if (waitKey(5) >= 0)
            break;
    }
    cap.release();  //释放捕获器
    return 0;
}