# OpenCV 2_图像采集

## 1. 图像数据的读取和输出

1. `imread`函数
```c++
Mat imread(const String & filename,int  flags=IMREAD_COLOR);
```
>`filename`：需要读取图像的文件名称，包含图像地址、名称和图像文件扩展名
>
>`flags`：读取图像形式的标志，如将彩色图像按照灰度图读取，默认参数是按照彩色图像格式读取，标志参数在功能不冲突的前提下可以同时声明多个，不同参数之间用“|”隔开

>如果图像文件不存在、破损或者格式不受支持时，则无法读取图像，此时函数返回一个空矩阵，因此可以通过**判断返回矩阵的`data`属性是否为空或者`empty()`函数是否为真**来判断是否成功读取图像，如果读取图像失败，`data`属性返回值为0，`empty()`函数返回值为1。

>无论在哪个系统中，bmp文件和dib文件都是始终可以读取的，在Windows和Mac系统中，默认情况下使用OpenCV自带的编解码器（libjpeg，libpng，libtiff和libjasper），因此可以读取JPEG（jpg、jpeg、jpe），PNG，TIFF（tiff、tif）文件，在Linux系统中需要自行安装这些编解码器，安装后同样可以读取这些类型的文件。
>>该函数能否读取文件数据与扩展名无关，而是通过文件的内容确定图像的类型。

>flags|值|含义
>-|-|-
>IMREAD_UNCHANGED|-1|按照图像原样读取
>IMREAD_GRAYSCALE|0|将图像转为单通道灰度图像后读取
>IMREAD_COLOR|1|将图像转换为3通道BGR彩色图像
>IMREAD_ANYDEPTH|2|保留原图像的16位，32位深度，不声明则改为8位读取
>IMREAD_ANYCOLOR|4|以任何颜色读取图像
>IMREAD_LOAD_GDAL|8|使用gdal驱动程序加载图像
>IMREAD_REDUCED_GRAYSCALE_2|16|将图像转换为单通道灰度图像，尺寸缩小1/2，更改最后一位数字可更改为缩小1/4和1/8。
>IMREAD_REDUCED_COLOR_2|17|将图像转换为三通道彩色图像，尺寸缩小1/2，更改最后一位数字可更改为缩小1/4和1/8。
>IMREAD_IGNORE_ORIENTATION|128|不以EXIF方向旋转图像

2. `imshow`函数
```c++
void imshow(const String & winname,InputArray mat);
```
>`winname`：要显示图像的窗口的名字，用字符串形式赋值
>
>`mat`：要显示的图像矩阵

该函数会在指定的窗口中显示图像，如果在此函数之前没有创建同名的图像窗口，就会以WINDOW_AUTOSIZE标志创建一个窗口，显示图像的原始大小，如果创建了图像窗口，则会缩放图像以适应窗口属性。该函数会根据图像的深度将其缩放，具体缩放规则为：

- 如果图像是8位无符号类型，则按照原样显示
- 如果图像是16位无符号类型或者32位整数类型，则会将像素除以256，将范围由[0,255*256]映射到[0,255]
- 如果图像时32位或64位浮点类型，则将像素乘以255，即将范围由[0,1]映射到[0,255]

>此函数运行后会继续执行后面程序，如果后面程序执行完直接退出的话，那么显示的图像有可能闪一下就消失了，因此在需要显示图像的程序中，**往往会在`imshow()`函数后跟有`waitKey()`函数，用于将程序暂停一段时间。`waitKey()`函数是以毫秒计的等待时长，如果参数缺省或者为“0”表示等待用户按键结束该函数。**

3. `namedWindow`函数
```c++
void cv::namedWindow(const String & winname,int  flags = WINDOW_AUTOSIZE);
```
>winname：窗口名称，用作窗口的标识符
>
>flags：窗口属性设置标志

该函数会创建一个窗口变量，用于显示图像和滑动条，通过窗口的名称引用该窗口，如果在创建窗口时已经存在具有相同名称的窗口，则该函数不会执行任何操作。创建一个窗口需要占用部分内存资源，因此通过该函数创建窗口后，在不需要窗口时需要关闭窗口来释放内存资源。

>OpenCV提供了两个关闭窗口资源的函数，分别是`destroyWindow()`函数和`destroyAllWindows()`，通过名称我们可以知道前一个函数是用于关闭一个指定名称的窗口，即在括号内输入窗口名称的字符串即可将对应窗口关闭，后一个函数是关闭程序中所有的窗口，一般用于程序的最后。

>flag|值|含义
>-|-|-
>WINDOW_NORMAL|0X0000 0000|显示图像后允许用户调整大小
>WINDOW_AUTOSIZE|0X0000 0001|根据图像大小显示窗口，不允许用户调整大小
>WINDOW_OPENGL|0X0000 1000|创建窗口时支持OpenGL
>WINDOW_FULLSCREEN|1|全屏显示窗口
>WINDOW_FREERATIO|0X0000 0100|调整图像尺寸以充满窗口|
>WINDOW_KEEPRATIO|0X0000 0000|保持图像比例
>WINDOW_GUIEXPANDED|0X0000 0000|创建窗口允许添加工具栏和状态栏
>WINDOW_GUI_NORMAL|0X0000 0010|创建没有工具栏和状态栏的窗口
## 2. 视频数据的读取与输出

1. 视频读取函数
```c++
VideoCapture :: VideoCapture(const String& filename,int apiPreference =CAP_ANY)
```
>filename：读取的视频文件或者图像序列名称
>
>apiPreference：读取数据时设置的属性

>可以读取的文件种类包括**视频文件(例如video.avi)、图像序列或者视频流的URL**。
>>其中读取图像序列需要将多个图像的名称统一为“前缀+数字”的形式，通过“前缀+%02d”的形式调用，例如在某个文件夹中有图片img_00.jpg、img_01.jpg、img_02.
jpg……加载时文件名用img_%02d.jpg表示。
>>
>>函数中的读取视频设置属性标签默认的是自动搜索合适的标志，所以在平时使用中，可以将其缺省，只需要输入视频名称即可。

>构造函数同样有可能读取文件失败，因此需要通过`isOpened()`函数进行判断，如果读取成功则返回值为true，如果读取失败，则返回值为false。

>需要使用视频中的图像时，还需要将图像由VideoCapture类变量里导出到Mat类变量里，用于后期数据处理，**该操作可以通过“>>”运算符将图像按照视频顺序由VideoCapture类变量复制给Mat类变量**，也可以使用`read()`函数。当VideoCapture类变量中所有的图像都赋值给Mat类变量后，再次赋值的时候Mat类变量会变为空矩阵，因此**可以通过`empty()`判断VideoCapture类变量中是否所有图像都已经读取完毕**

>VideoCapture类变量同时提供了可以查看视频属性的get()函数，通过输入指定的标志来获取视频属性，例如视频的像素尺寸、帧数、帧率等

>标志参数|值|含义
>-|-|-
>CAP_PROP_POS_MESC|0|视频文件的当前位置（单位为ms）
>CAP_PROP_FRAME_WIDTH|3|视频流中图像的宽度
>CAP_PROP_FRAME_HEIGHT|4|视频流中图像的高度
>CAP_PROP_FPS|5|视频流中图像帧率（每秒帧数）
>CAP_PROP_FOURCC|6|解编码器的4字符编码
>CAP_PROP_FRAME_COUNT|7|视频流中图像的帧数
>CAP_PROP_FORMAT|8|返回的Mat对象的格式
>CAP_PROP_BRIGHTNESS|10|图像的亮度（适用于相机）
>CAP_PROP_CONTRAST|11|图像的对比度
>CAP_PROP_SATURATION|12|图像的饱和度
>CAP_PROP_HUE|13|图像的色调
>CAP_PROP_GAIN|14|图像的增益
```c++
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    cout << "OpenCv Version: " << CV_VERSION << endl;
    //读取视频文件
    VideoCapture video("Project.mp4");
    //检验是否读取成功
    if (video.isOpened()) {
        cout << "视频中图像的宽度 = " << video.get(CAP_PROP_FRAME_WIDTH) << endl;
        cout << "视频中图像的高度 = " << video.get(CAP_PROP_FRAME_HEIGHT) << endl;
        cout << "视频帧率 = " << video.get(CAP_PROP_FPS) << endl;
        cout << "视频的总帧数 = " << video.get(CAP_PROP_FRAME_COUNT) << endl;
    }
    else {
        cout << "请确认视频文件名称是否正确" << endl;
        return -1;
    }
    //将视频中每一帧图片进行输出
    while (1) 
    {
        Mat frame;
        video >> frame;
        if (frame.empty()) 
        {
            break;
        }
        imshow("video", frame);
        waitKey(1000 / video.get(CAP_PROP_FPS));
    }
    waitKey();
    return 0;
}
```

## 3. 图像和视频的保存
1. `imwrite()`函数
```c++
bool imwrite(const String& filename,InputArray img,Const std::vector<int>& params = std::vector<int>());
```
>filename：保存图像的地址和文件名，包含图像格式
>
>img：将要保存的Mat类矩阵变量
>
>params：保存图片格式属性设置标志

该函数用于将Mat类矩阵保存成图像文件，如果成功保存，则返回true，否则返回false。可以保存的图像格式参考imread()函数能够读取的图像文件格式，通常使用该函数只能保存8位单通道图像和3通道BGR彩色图像，但是可以通过更改第三个参数保存成不同格式的图像。
>16位无符号（CV_16U）图像可以保存成PNG、JPEG、TIFF格式文件；
>
>32位浮点（CV_32F）图像可以保存成PFM、TIFF、OpenEXR和Radiance HDR格式文件；
>
>4通道（Alpha通道）图像可以保存成PNG格式文件。

第三个参数设置方式如下：
```c++
vector <int> compression_params;
compression_params.push_back(IMWRITE_PNG_COMPRESSION);
compression_params.push_back(9);
imwrite(filename, img, compression_params);
```

>标志参数|值|作用
>-|-|-
>IMWRITE_JPEG_QUALITY|1|保存成JPEG格式的图像文件质量，0-100，默认95
>IMWRITE_JPEG_PROGRESSIVE|2|增强JPEG格式，启用为1，默认为0
>IMWRITE_JPEG_OPTIMIZE|3|优化JPEG格式，启用为1，默认为0
>IMWRITE_JPEG_LUMA_QUALITY|5|JPEG文件亮度质量等级，0-100，默认为0
>IMWRITE_JPEG_CHROMA_QUALITY|6|JPEG文件色度质量等级|0-100，默认为0
>IMWRITE_PNG_COMPRESSION|16|保存成PNG文件压缩级别，0-0，值越高尺寸更小，压缩时间更长
>IMWRITE_TIFF_COMPRESSION|25|保存为TIFF格式压缩方案

```c++
#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void CreateMat(Mat& mat)
{
	CV_Assert(mat.channels() == 4);
	for (int i = 0; i < mat.rows; ++i)
	{
		for (int j = 0; j < mat.cols; ++j)
		{
			Vec4b& channel = mat.at<Vec4b>(i, j);
			channel[0] = UCHAR_MAX;
			channel[1] = saturate_cast<uchar>((float(mat.cols - j)) / ((float)mat.cols) * UCHAR_MAX);
			channel[2] = saturate_cast<uchar>((float(mat.rows - i)) / ((float)mat.rows) * UCHAR_MAX);
			channel[3] = saturate_cast<uchar>(0.5 * (channel[1] + channel[2]));
		}
	}
}

int main()
{
	cout << "OpenCV Version: " << CV_VERSION << endl;
	Mat mat(480, 640, CV_8UC4, Scalar(0, 0, 0, 0));

	CreateMat(mat);

	vector<int>comprocession_params;
	comprocession_params.push_back(IMWRITE_PNG_COMPRESSION);
	comprocession_params.push_back(9);

	bool result = imwrite("picture.png", mat, comprocession_params);

	if (!result)
	{
		cout << "failed!" << endl;
		return -1;
	}
	cout << "Success!" << endl;
	imshow("picture.png", mat);
	waitKey(0);
	return 0;
}
```

2. `videowriter()`函数
```c++
VideoWriter :: VideoWriter(const String& filename,
                                       int fourcc,
                                       double  fps,
                                       Size frameSize,
                                       bool  isColor=true
                                       )
```
>`filename`：保存视频的地址和文件名，包含视频格式
>
>`fourcc`：压缩帧的4字符编解码器代码
>
>`fps`：保存视频的帧率，即视频中每秒图像的张数。
>
>`framSize`：视频帧的尺寸
>
>`isColor`：保存视频是否为彩色视频

默认构造函数的使用方法与VideoCapture()相同，都是创建一个用于保存视频的数据流，后续通过`open()`函数设置保存文件名称、编解码器、帧数等一系列参数。

可以通过`isOpened()`函数判断是否成功创建一个视频流，可以通过`get()`查看视频流中的各种属性。在保存视频时，我们只需要将生成视频的图像一帧一帧通过“<<”操作符（或者`write()`函数）赋值给视频流即可，最后使用`release()`关闭视频流。

>解编码器标志|含义
>-|-
>CV_ROURCC('D','I','V','X')|MPEG-4编码
>CV_ROURCC('P','I','M','1')|MPEG-1编码
>CV_ROURCC('M','J','P','G')|JPEG编码
>CV_ROURCC('M','P','4','2')|MPEG-4.2编码
>CV_ROURCC('D','I','V','3')|MPEG-4.3编码
>CV_ROURCC('U','2','6','3')|H263编码
>CV_ROURCC('I','2','6','3')|H263I编码
>CV_ROURCC('F','L','V','1')|FLV1编码

```c++
#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
//视频加载
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
//视频以帧格式保存在mat对象之中
	Mat mat;

	video >> mat;
		
	if (mat.empty())
	{
		cout << "No image!" << endl;
	}

//视频流创建和配置
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
//将读取的视频帧写入视频流中并保存

	while (1)
	{
		if (!video.read(mat))
		{
			cout << "读取完毕" << endl;
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
//释放对象
	video.release();
	writer.release();
	return 0;

}
```

## 4. 保存和读取XML和YMAL文件

XML是一种元标记语言，所谓元标记就是使用者可以根据自身需求定义自己的标记，XML是一种结构化的语言，通过XML语言可以知道数据之间的隶属关系。通过标记的方式，无论以任何形式保存数据，只要文件满足XML格式，那么读取出来的数据就不会出现混淆和歧义。XML文件的扩展名是“.xml”。

YMAL是一种以数据为中心的语言，通过“变量:数值”的形式来表示每个数据的数值，通过不同的缩进来表示不同数据之间的结构和隶属关系。YMAL可读性高，常用来表达资料序列的格式，它参考了多种语言，包括XML、C语言、Python、Perl等。YMAL文件的扩展名是“.ymal”或者“.yml”。

1. `FileStorage`构造函数
```c++
FileStorage::FileStorage(const String & filename,
                         int  flags,
                         const String & encoding = String()
                        );
```
>filename：打开的文件名称。
>
>flags：对文件进行的操作类型标志。
>
>encodin：编码格式，目前不支持UTF-16 XML编码，需要使用UTF-8 XML编码。

>flags|值|含义
>-|-|-
>READ|0|读取文件中的数据
>WRITE|1|向文件中重新写入数据并覆盖之前的数据
>APPEND|2|向文件中继续写入数据，新数据在原数据之后
>MEMORY|4|将数据读取到内部缓冲区

打开文件后，可以通过FileStorage类中的`isOpened()`函数判断是否成功打开文件，如果成功打开文件，该函数返回true，如果打开文件失败，则该函数返回false。

2. `open()`函数
```c++
virtual bool cv::FileStorage::open(const String & filename,
                                   int  flags,
                                   const String & encoding = String()
                                  );
```

打开文件后，类似C++中创建的数据流，可以通过“<<”操作符将数据写入文件中，或者通过“>>”操作符从文件中读取数据。

3. `write()`函数
```c++
void cv::FileStorage::write(const String & name,int  val);
```
>name：写入文件中的变量名称。
>
>val：变量值。

- 使用操作符向文件中写入数据时与write()函数类似，都需要声明变量名和变量值，例如变量名为“age”，变量值为“24”，可以通过“file<<”age”<<24”来实现。
>如果某个变量的数据是一个数组，**可以用“[]”将属于同一个变量的变量值标记出来，例如“file<<”age”<<“[”<<24<<25<<”]””**。
>
>如果某些变量隶属于某个变量，可以用“{}”表示变量之间的隶属关系，例如“file<<”age”<<“{”<<”Xiaoming”<<24<<”Wanghua”<<25<<”}””。

- 读取文件中的数据时，只需要通过变量名就可以读取变量值。例如“file [“x”] >> xRead”是读取变量名为x的变量值。
>但是，当某个变量中含有多个数据或者含有子变量时，就需要通过`FileNode`节点类型和迭代器`FileNodeIterator`进行读取，例如某个变量的变量值是一个数组，首先需要
定义一个file [“age”]的FileNode节点类型变量，之后通过迭代器遍历其中的数据。
>
>另外一种方法可以不使用迭代器，**通过在变量后边添加“[]”地址的形式读取数据**，例如FileNode[0]表示数组变量中的第一个数据，FileNode[“Xiaoming”]表示“age”变量中的“Xiaoming”变量的数据，依次向后添加“[]”地址实现多节点数据的读取。

```c++
#include<iostream>
#include<vector>
#include<string>
//#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
	cout << "OpenCv Version: " << CV_VERSION << endl;
	FileStorage file;
	string filename = "text.yaml";

	file.open(filename, FileStorage::WRITE);

	if (!file.isOpened())
	{
		cout << "Fail to open!" << endl;
		return -1;
	}

	Mat mat = Mat::eye(3, 3, CV_8U);

	file.write("Mat", mat);

	float x = 100;
	file << "x" << x;

	string str = "Test";
	file << "str" << str;

	file <<  "Array" << "[" << 1 << 2 << 3 << 4 << "]";

	file << "Time" << "{" << "year" << 2022 << "month" << 2 << "day" << 17 << "d_time" << "[" << 0 << 1 << 2 << 3 << "]" << "}";

	file.release();

	FileStorage rfile;
	rfile.open(filename, FileStorage::READ);

	if (!rfile.isOpened())
	{
		cout << "Fail to open!" << endl;
		return -1;
	}

	float x_r;
	rfile["x"] >> x_r;
	cout << "x_r = " << x_r << endl;

	string str_r;
	rfile["str"] >> str_r;
	cout << "str_r = " << str_r << endl;

	FileNode node = rfile["Array"];
	cout << "Array = [";
	for (FileNodeIterator it = node.begin(); it != node.end(); it++)
	{
		int i;
		*it >> i;
		cout << i << " ";
	}
	cout << "]" << endl;

	Mat mat_r;
	rfile["Mat"] >> mat_r;
	cout << "mat_r = " << mat_r << endl;

	FileNode t_node = rfile["Time"];
	int year_r = (int)t_node["year"];
	int month_r = (int)t_node["month"];
	int day_r = (int)t_node["day"];
	int hour_r = (int)t_node["d_time"][0];
	int minute_r = (int)t_node["d_time"][1];
	int second_r = (int)t_node["d_time"][2];

	cout << "Time: " << year_r << "." << month_r << "." << day_r << "." << hour_r << ":" << minute_r << ":" << second_r << endl;


	rfile.release();
	system("pause");
	return 0;
}
```