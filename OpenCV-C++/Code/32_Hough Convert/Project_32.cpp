#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

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

void drawLIne(Mat& img,                 //要标记的图像
    vector<Vec2f>lines,                 //检测的直线数据
    double rows,                        //原图像的行数
    double cols,                        //原图像的列数
    Scalar scalar,                      //绘制直线的颜色
    int n                               //绘制直线的线宽
)
{
    Point pt1, pt2;
    for (size_t i = 0; i < lines.size(); ++i) 
    {
        float rho = lines[i][0];                    //直线距离坐标原点的距离
        float theta = lines[i][1];                  //直线过坐标原点垂线与x轴夹角
        double a = cos(theta);                      //夹角的余弦值
        double b = sin(theta);                      //夹角的正弦值
        double x0 = a * rho, y0 = b * rho;          //直线与坐标原点垂线的交点
        double length = max(rows, cols);            //图像高宽的最大值

        //计算直线上的一点
        pt1.x = cvRound(x0 + length * (-b));
        pt1.y = cvRound(y0 + length * (a));
        //计算直线上另一点
        pt2.x = cvRound(x0 - length * (-b));
        pt2.y = cvRound(y0 - length * (a));
        //两点绘制一条直线
        line(img, pt1, pt2, scalar, n);
    }
}


int main()
{
	Mat img,edge;
	Read_Img(img, "picture.jpeg");

	Canny(img, edge, 50, 200);
	threshold(edge, edge, 170, 255, THRESH_BINARY);

	vector<Vec2f>lines;
	HoughLines(edge, lines, 1, CV_PI / 100, 50, 0, 0);

    Mat imgl;
    img.copyTo(imgl);
    drawLIne(imgl, lines, edge.rows, edge.cols, Scalar(255), 2);

    imshow("edge", edge);
    imshow("img", img);
    imshow("img1", imgl);
    waitKey(0);
    return 0;

}