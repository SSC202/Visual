#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	VideoCapture video = VideoCapture("Project.mp4");

	if (video.isOpened())
	{
		cout << "Width = " << video.get(CAP_PROP_FRAME_WIDTH);
		cout << "Height = " << video.get(CAP_PROP_FRAME_HEIGHT);
		cout << "FPS = " << video.get(CAP_PROP_FPS);
	}
	else
	{
		cout << "Failed!" << endl;
		return -1;
	}

	Mat mat;

	video >> mat;
		
	if (mat.empty())
	{
		cout << "No image!" << endl;
	}


	bool isColor = (mat.type() == CV_8UC3);
	VideoWriter writer;
	int coder = VideoWriter::fourcc('M', 'J', 'P', 'G');

	double fps = 50.0;
	string filename = "product.avi";
	writer.open(filename, coder, fps, mat.size(), isColor);
	if (!writer.isOpened())
	{
		cout << "ERROR!" << endl;
		return -1;
	}

	while (1)
	{
		if (!video.read(mat))
		{
			cout << "¶ÁÈ¡Íê±Ï" << endl;
			break;
		}
		writer.write(mat);
		imshow("product", mat);

		char c = waitKey(50);
		if (c == 27)
		{
			break;
		}
	}

	video.release();
	writer.release();
	return 0;

}