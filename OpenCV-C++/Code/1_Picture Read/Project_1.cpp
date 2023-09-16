#include<opencv2\opencv.hpp>

using namespace cv;

int main(int argc, char** argv)
{
	//下载图片
	Mat src = imread("picture_1.jpg", IMREAD_COLOR);
	if (src.empty())
	{
		printf("could not load image ...\n");
		return -1;
	}

	namedWindow("Test");
	imshow("Test", src);
	waitKey(20);

	//转换色彩空间
	namedWindow("Transform Test");
	Mat t_src;
	cvtColor(src, t_src, COLOR_BGR2HLS);
	imshow("Transform Test", t_src);
	waitKey(0);

	//转换图片另存为文件
	imwrite("t_picture_1.jpg", t_src);


	return 0;
}