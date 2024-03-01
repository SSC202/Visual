#include<opencv2\opencv.hpp>

using namespace cv;

int main(int argc, char** argv)
{
	// ����ͼƬ
	Mat src = imread("picture.jpg", IMREAD_COLOR);
	if (src.empty())
	{
		printf("could not load image ...\n");
		return -1;
	}

	namedWindow("Test");
	imshow("Test", src);
	waitKey(20);

	// ת��ɫ�ʿռ�
	namedWindow("Transform Test");
	Mat t_src;
	cvtColor(src, t_src, COLOR_BGR2HLS);
	imshow("Transform Test", t_src);
	waitKey(0);

	// ת��ͼƬ���Ϊ�ļ�
	imwrite("t_picture_1.jpg", t_src);

	return 0;
}