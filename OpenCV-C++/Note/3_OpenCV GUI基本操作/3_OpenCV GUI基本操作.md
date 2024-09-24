# OpenCV C++ 3_GUI基本操作

## 1. 图像的读取，显示和保存

### 图像的读取

```c++
/**
  * @brief	图像读取函数
  * @param	filename	图像文件路径
  * @param	flag		读取方式
  */
Mat imread(const String & filename,int flags = IMREAD_COLOR);
```

如果图像文件不存在、破损或者格式不受支持时，则无法读取图像，此时函数返回一个空矩阵，因此可以通过判断返回矩阵的`data`属性是否为空或者`empty()`函数是否为真来判断是否成功读取图像，如果读取图像失败，`data`属性返回值为0，`empty()`函数返回值为1。

无论在哪个系统中，`bmp`文件和`dib`文件都是始终可以读取的，在Windows和Mac系统中，默认情况下使用OpenCV自带的编解码器（`libjpeg`，`libpng`，`libtiff`和`libjasper`），因此可以读取`JPEG`（`jpg`、`jpeg`、`jpe`），`PNG`，`TIFF`（`tiff`、`tif`）文件，在Linux系统中需要自行安装这些编解码器，安装后同样可以读取这些类型的文件。

>该函数能否读取文件数据与扩展名无关，而是通过文件的内容确定图像的类型。

| `flag`                       | 值   | 含义                                                         |
| ---------------------------- | ---- | ------------------------------------------------------------ |
| `IMREAD_UNCHANGED`           | -1   | 按照图像原样读取                                             |
| `IMREAD_GRAYSCALE`           | 0    | 将图像转为单通道灰度图像后读取                               |
| `IMREAD_COLOR`               | 1    | 将图像转换为3通道`BGR`彩色图像                               |
| `IMREAD_ANYDEPTH`            | 2    | 保留原图像的16位，32位深度，不声明则改为8位读取              |
| `IMREAD_ANYCOLOR`            | 4    | 以任何颜色读取图像                                           |
| `IMREAD_LOAD_GDAL`           | 8    | 使用gdal驱动程序加载图像                                     |
| `IMREAD_REDUCED_GRAYSCALE_2` | 16   | 将图像转换为单通道灰度图像，尺寸缩小1/2，更改最后一位数字可更改为缩小1/4和1/8。 |
| `IMREAD_REDUCED_COLOR_2`     | 17   | 将图像转换为三通道彩色图像，尺寸缩小1/2，更改最后一位数字可更改为缩小1/4和1/8。 |
| `IMREAD_IGNORE_ORIENTATION`  | 128  | 不以`EXIF`方向旋转图像                                       |

### 图像的显示

```c++
/**
  * @brief	图像显示函数
  * @param	winname	窗口名称，不同的窗口应有不同的名称
  * @param	mat		图像，窗口生成后自动调整为图像大小
  */
void imshow(const String & winname,InputArray mat);
```

该函数会在指定的窗口中显示图像，如果在此函数之前没有创建同名的图像窗口，就会以`WINDOW_AUTOSIZE`标志创建一个窗口，显示图像的原始大小，如果创建了图像窗口，则会缩放图像以适应窗口属性。该函数会根据图像的深度将其缩放，具体缩放规则为：

- 如果图像是8位无符号类型，则按照原样显示
- 如果图像是16位无符号类型或者32位整数类型，则会将像素除以256，将范围由[0,255*256]映射到[0,255]
- 如果图像时32位或64位浮点类型，则将像素乘以255，即将范围由[0,1]映射到[0,255]

>此函数运行后会继续执行后面程序，如果后面程序执行完直接退出的话，那么显示的图像有可能闪一下就消失了，因此在需要显示图像的程序中，往往会在`imshow()`函数后跟有`waitKey()`函数，用于将程序暂停一段时间。`waitKey()`函数是以毫秒计的等待时长，检查是否有键盘输入，若按下键盘，则返回对应的ASCII值，程序继续运行；如果无输入则返回-1，若此值取0，则等待时间为无限期。

```c++
/**
  * @brief	窗口显示函数
  * @param	winname	窗口名称，不同的窗口应有不同的名称
  * @param	flags 	窗口属性设置标志
  */
void cv::namedWindow(const String & winname,int flags = WINDOW_AUTOSIZE);
```

该函数会创建一个窗口变量，用于显示图像和滑动条，通过窗口的名称引用该窗口，如果在创建窗口时已经存在具有相同名称的窗口，则该函数不会执行任何操作。创建一个窗口需要占用部分内存资源，因此通过该函数创建窗口后，在不需要窗口时需要关闭窗口来释放内存资源。

>OpenCV提供了两个关闭窗口资源的函数，分别是`destroyWindow()`函数和`destroyAllWindows()`，通过名称我们可以知道前一个函数是用于关闭一个指定名称的窗口，即在括号内输入窗口名称的字符串即可将对应窗口关闭，后一个函数是关闭程序中所有的窗口，一般用于程序的最后。

| `flag`               | 值            | 含义                                     |
| -------------------- | ------------- | ---------------------------------------- |
| `WINDOW_NORMAL`      | `0X0000 0000` | 显示图像后允许用户调整大小               |
| `WINDOW_AUTOSIZE`    | `0X0000 0001` | 根据图像大小显示窗口，不允许用户调整大小 |
| `WINDOW_OPENGL`      | `0X0000 1000` | 创建窗口时支持OpenGL                     |
| `WINDOW_FULLSCREEN`  | `1`           | 全屏显示窗口                             |
| `WINDOW_FREERATIO`   | `0X0000 0100` | 调整图像尺寸以充满窗口                   |
| `WINDOW_KEEPRATIO`   | `0X0000 0000` | 保持图像比例                             |
| `WINDOW_GUIEXPANDED` | `0X0000 0000` | 创建窗口允许添加工具栏和状态栏           |
| `WINDOW_GUI_NORMAL`  | `0X0000 0010` | 创建没有工具栏和状态栏的窗口             |

### 图像的保存

```c++
/**
  * @brief	图像保存函数
  * @param	filename 	保存图像的地址和文件名，包含图像格式
  * @param	img 		将要保存的Mat类矩阵变量
  * @param 	params 		保存图片格式属性设置标志
  */
bool imwrite(const String& filename,InputArray img,Const std::vector<int>& params = std::vector<int>());
```

该函数用于将Mat类矩阵保存成图像文件，如果成功保存，则返回true，否则返回false。可以保存的图像格式参考imread()函数能够读取的图像文件格式，通常使用该函数只能保存8位单通道图像和3通道BGR彩色图像，但是可以通过更改第三个参数保存成不同格式的图像。

>16位无符号（`CV_16U`）图像可以保存成`PNG`、`JPEG`、`TIFF`格式文件；
>
>32位浮点（`CV_32F`）图像可以保存成`PFM`、`TIFF`、`OpenEXR`和`Radiance HDR`格式文件；
>
>4通道（Alpha通道）图像可以保存成`PNG`格式文件。

第三个参数设置方式如下：

```c++
vector <int> compression_params;
compression_params.push_back(IMWRITE_PNG_COMPRESSION);
compression_params.push_back(9);

imwrite(filename, img, compression_params);
```

>| 标志参数                      | 值   | 作用                                                       |
>| ----------------------------- | ---- | ---------------------------------------------------------- |
>| `IMWRITE_JPEG_QUALITY`        | 1    | 保存成`JPEG`格式的图像文件质量，0-100，默认95              |
>| `IMWRITE_JPEG_PROGRESSIVE`    | 2    | 增强`JPEG`格式，启用为1，默认为0                           |
>| `IMWRITE_JPEG_OPTIMIZE`       | 3    | 优化`JPEG`格式，启用为1，默认为0                           |
>| `IMWRITE_JPEG_LUMA_QUALITY`   | 5    | `JPEG`文件亮度质量等级，0-100，默认为0                     |
>| `IMWRITE_JPEG_CHROMA_QUALITY` | 6    | `JPEG`文件色度质量等级                                     |
>| `IMWRITE_PNG_COMPRESSION`     | 16   | 保存成`PNG`文件压缩级别，0-0，值越高尺寸更小，压缩时间更长 |
>| `IMWRITE_TIFF_COMPRESSION`    | 25   | 保存为`TIFF`格式压缩方案                                   |

## 2. 视频的获取和保存

### 视频的获取

视频是`VideoCapture`对象，其参数可以为摄像头设备的索引号，或者一个视频文件。

笔记本电脑的内置摄像头对应的参数就是0。可以改变此值选择别的摄像头。

```c++
/**
  * @brief	视频对象创建保存函数
  * @param	filename 		读取的视频文件或者图像序列名称
  * @param	apiPreference 	读取数据时设置的属性
  */
VideoCapture :: VideoCapture(const String& filename,int apiPreference =CAP_ANY)
```

1. 可以读取的文件种类包括视频文件、图像序列或者视频流的URL。

   其中读取图像序列需要将多个图像的名称统一为`前缀+数字`的形式，通过`前缀+%02d`的形式调用，例如在某个文件夹中有图片`img_00.jpg`、`img_01.jpg`、`img_02.jpg`……加载时文件名用`img_%02d.jpg`表示。

   函数中的读取视频设置属性标签默认的是自动搜索合适的标志，所以在平时使用中，可以将其缺省，只需要输入视频名称即可。

2. 构造函数同样有可能读取文件失败，因此需要通过`isOpened()`函数进行判断，如果读取成功则返回值为`true`，如果读取失败，则返回值为`false`。

3. 需要使用视频中的图像时，还需要将图像由`VideoCapture`类变量里导出到Mat类变量里，用于后期数据处理，该操作可以通过“>>”运算符将图像按照视频顺序由`VideoCapture`类变量复制给`Mat`类变量，也可以使用`read()`函数。当`VideoCapture`类变量中所有的图像都赋值给`Mat`类变量后，再次赋值的时候Mat类变量会变为空矩阵，因此可以通过`empty()`判断`VideoCapture`类变量中是否所有图像都已经读取完毕。

4. `VideoCapture`类变量同时提供了可以查看视频属性的`get()`函数，通过输入指定的标志来获取视频属性，例如视频的像素尺寸、帧数、帧率等


| 标志参数                | 值   | 含义                           |
| ----------------------- | ---- | ------------------------------ |
| `CAP_PROP_POS_MESC`     | 0    | 视频文件的当前位置（单位为ms） |
| `CAP_PROP_FRAME_WIDTH`  | 3    | 视频流中图像的宽度             |
| `CAP_PROP_FRAME_HEIGHT` | 4    | 视频流中图像的高度             |
| `CAP_PROP_FPS`          | 5    | 视频流中图像帧率（每秒帧数）   |
| `CAP_PROP_FOURCC`       | 6    | 解编码器的4字符编码            |
| `CAP_PROP_FRAME_COUNT`  | 7    | 视频流中图像的帧数             |
| `CAP_PROP_FORMAT`       | 8    | 返回的Mat对象的格式            |
| `CAP_PROP_BRIGHTNESS`   | 10   | 图像的亮度（适用于相机）       |
| `CAP_PROP_CONTRAST`     | 11   | 图像的对比度                   |
| `CAP_PROP_SATURATION`   | 12   | 图像的饱和度                   |
| `CAP_PROP_HUE`          | 13   | 图像的色调                     |
| `CAP_PROP_GAIN`         | 14   | 图像的增益                     |

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

### 视频的保存

为对每一帧图像处理后得到保存，需要创建`VideoWriter`对象

```c++
/**
  * @brief	VideoWriter对象创建
  * @param	filename 		输出文件名
  * @param	apiPreference 	编码格式
  * @param  fps				播放频率
  * @param  frameSize		帧的大小
  */
VideoWriter :: VideoWriter(const String& filename,
                                       int fourcc,
                                       double  fps,
                                       Size frameSize,
                                       bool  isColor=true
                                       )
```

默认构造函数的使用方法与`VideoCapture()`相同，都是创建一个用于保存视频的数据流，后续通过`open()`函数设置保存文件名称、编解码器、帧数等一系列参数。

可以通过`isOpened()`函数判断是否成功创建一个视频流，可以通过`get()`查看视频流中的各种属性。在保存视频时，我们只需要将生成视频的图像一帧一帧通过`<<`操作符（或者`write()`函数）赋值给视频流即可，最后使用`release()`关闭视频流。

| 解编码器标志                 | 含义           |
| ---------------------------- | -------------- |
| `CV_ROURCC('D','I','V','X')` | `MPEG-4`编码   |
| `CV_ROURCC('P','I','M','1')` | `MPEG-1`编码   |
| `CV_ROURCC('M','J','P','G')` | `JPEG`编码     |
| `CV_ROURCC('M','P','4','2')` | `MPEG-4.2`编码 |
| `CV_ROURCC('D','I','V','3')` | `MPEG-4.3`编码 |
| `CV_ROURCC('U','2','6','3')` | `H263`编码     |
| `CV_ROURCC('I','2','6','3')` | `H263I`编码    |
| `CV_ROURCC('F','L','V','1')` | `FLV1`编码     |

## 3. `XML`和`YMAL`文件的读取和保存

`XML`是一种元标记语言，所谓元标记就是使用者可以根据自身需求定义自己的标记，`XML`是一种结构化的语言，通过`XML`语言可以知道数据之间的隶属关系。通过标记的方式，无论以任何形式保存数据，只要文件满足`XML`格式，那么读取出来的数据就不会出现混淆和歧义。`XML`文件的扩展名是`.xml`。

`YMAL`是一种以数据为中心的语言，通过“变量:数值”的形式来表示每个数据的数值，通过不同的缩进来表示不同数据之间的结构和隶属关系。`YMAL`可读性高，常用来表达资料序列的格式，它参考了多种语言，包括XML、C语言、Python、Perl等。`YMAL`文件的扩展名是`.ymal`或者`.yml`。

1. `FileStorage`构造函数
```c++
/**
  * @brief	FileStorage 构造函数
  * @param  filename 	打开的文件名称 
  * @param  flags 		对文件进行的操作类型标志
  * @param  encoding	编码格式，目前不支持UTF-16 XML编码，需要使用UTF-8 XML编码
  */
FileStorage::FileStorage(const String & filename,
                         int  flags,
                         const String & encoding = String()
                        );
```
>flags|值|含义
>-|-|-
>`READ`|0|读取文件中的数据
>`WRITE`|1|向文件中重新写入数据并覆盖之前的数据
>`APPEND`|2|向文件中继续写入数据，新数据在原数据之后
>`MEMORY`|4|将数据读取到内部缓冲区

打开文件后，可以通过`FileStorage`类中的`isOpened()`函数判断是否成功打开文件，如果成功打开文件，该函数返回true，如果打开文件失败，则该函数返回false。

```c++
/**
  * @brief	文件打开函数
  * @param  filename 	打开的文件名称 
  * @param  flags 		对文件进行的操作类型标志
  * @param  encoding	编码格式，目前不支持UTF-16 XML编码，需要使用UTF-8 XML编码
  */
virtual bool cv::FileStorage::open(const String & filename,
                                   int  flags,
                                   const String & encoding = String()
                                  );
```

打开文件后，类似C++中创建的数据流，可以通过`<<`操作符将数据写入文件中，或者通过`>>`操作符从文件中读取数据。

```c++
/**
  * @brief	文件写入函数
  * @param  name 		写入文件中的变量名称
  * @param  val 		变量值
  */
void cv::FileStorage::write(const String & name,int  val);
```
- 使用操作符向文件中写入数据时与`write()`函数类似，都需要声明变量名和变量值，例如变量名为“age”，变量值为“24”，可以通过“file<<”age”<<24”来实现。
>如果某个变量的数据是一个数组，可以用`[]`将属于同一个变量的变量值标记出来，例如`file<<”age”<<“[”<<24<<25<<”]”`。
>
>如果某些变量隶属于某个变量，可以用`{}`表示变量之间的隶属关系，例如`file<<”age”<<“{”<<”Xiaoming”<<24<<”Wanghua”<<25<<”}”`。

- 读取文件中的数据时，只需要通过变量名就可以读取变量值。例如`file [“x”] >> xRead`是读取变量名为x的变量值。
>但是，当某个变量中含有多个数据或者含有子变量时，就需要通过`FileNode`节点类型和迭代器`FileNodeIterator`进行读取，例如某个变量的变量值是一个数组，首先需要定义一个`file [“age”]`的`FileNode`节点类型变量，之后通过迭代器遍历其中的数据。

>另外一种方法可以不使用迭代器，通过在变量后边添加`[]`地址的形式读取数据，例如`FileNode[0]`表示数组变量中的第一个数据，`FileNode[“Xiaoming”]`表示`“age”`变量中的`“Xiaoming”`变量的数据，依次向后添加`[]`地址实现多节点数据的读取。

## 4. 绘图函数

绘图统一参数：

> 1. `img`：绘图用的图像
> 2. `color`：绘图的颜色，RGB图为一个元组，灰度图给出灰度值。
> 3. `thickness`：绘图的粗细，默认为1，如果为-1则为闭合填充。
> 4. `linetype`：线条类型，默认为8连接，`cv2.LINE_AA`为抗锯齿，图像会比较平滑。

- 线条函数

```c++
/**
  * @brief	线条绘制函数
  * @param  img			图像
  * @param  pt1			左上角点(Point(x,y))
  * @param	pt2			右下角点(Point(x,y))
  */
void line(InputOutputArray img, Point pt1, Point pt2, const Scalar& color,
                     int thickness = 1, int lineType = LINE_8, int shift = 0);
```

- 矩形函数

```c++
/**
  * @brief	矩形绘制函数
  * @param  img			图像
  * @param  pt1			左上角点(Point(x,y))
  * @param	pt2			右下角点(Point(x,y))
  */
void rectangle(InputOutputArray img, Point pt1, Point pt2,const Scalar& color, int thickness = 1,int lineType = LINE_8, int shift = 0);
```

- 圆形函数

```c++
/**
  * @brief	圆形绘制函数
  * @param  img			图像
  * @param  center		圆心坐标
  * @param	radius		半径
  */
void circle(InputOutputArray img, Point center, int radius,const Scalar& color, int thickness = 1,int lineType = LINE_8, int shift = 0);
```

- 椭圆函数

```c++
/**
  * @brief	椭圆绘制函数
  * @param  img			图像
  * @param  center		中心点坐标
  * @param	axes		两个轴的长度
  * @param	angle		椭圆沿逆时针旋转的角度
  * @param	startAngle	 椭圆弧的起始角度
  * @param	endAngle	椭圆弧的结束角度
  */
void ellipse(InputOutputArray img, Point center, Size axes,double angle, double startAngle, double endAngle,const Scalar& color, int thickness = 1,int lineType = LINE_8, int shift = 0);
```

- 多边形

对于多边形，需要指定每个顶点的坐标来构建一个数组。

```c++
/**
  * @brief	多边形绘制函数
  * @param  img			图像
  * @param  pts			点集
  * @param	isClosed	布尔值
  */
void polylines(InputOutputArray img, InputArrayOfArrays pts,bool isClosed, const Scalar& color,int thickness = 1, int lineType = LINE_8, int shift = 0 );
```

- 写文字

```c++
/**
  * @brief	文字绘制函数
  * @param  img			图像
  * @param  text		文字
  * @param  org		 	绘制位置
  * @param	fontFace	字体
  * @param  fontScale	字体大小
  */
void putText( InputOutputArray img, const String& text, Point org,int fontFace, double fontScale, Scalar color,int thickness = 1, int lineType = LINE_8,bool bottomLeftOrigin = false );
```

## 5. 鼠标事件

鼠标事件可以是鼠标上发生的任何动作，可以通过各种鼠标事件执行不同的任务。

```c++
EVENT_MOUSEMOVE              //滑动
EVENT_LBUTTONDOWN            //左键点击
EVENT_RBUTTONDOWN            //右键点击
EVENT_MBUTTONDOWN            //中键点击
EVENT_LBUTTONUP              //左键放开
EVENT_RBUTTONUP              //右键放开
EVENT_MBUTTONUP              //中键放开
EVENT_LBUTTONDBLCLK          //左键双击
EVENT_RBUTTONDBLCLK          //右键双击
EVENT_MBUTTONDBLCLK          //中键双击

EVENT_FLAG_LBUTTON        //左鍵拖曳
EVENT_FLAG_RBUTTON        //右鍵拖曳
EVENT_FLAG_MBUTTON        //中鍵拖曳
EVENT_FLAG_CTRLKEY        //(8~15)按Ctrl不放事件
EVENT_FLAG_SHIFTKEY      //(16~31)按Shift不放事件
EVENT_FLAG_ALTKEY        //(32~39)按Alt不放事件
```

鼠标事件发生后会调用对应的回调函数，从而执行对应的操作。

```c++
/**
  * @brief	鼠标事件设置函数
  * @param  winname 	窗口的名字
  * @param  onMouse 	鼠标响应回调函数。指定窗口里每次鼠标时间发生的时候，被调用的函数指针。
  						这个函数的原型应该为void on_Mouse(int event, int x, int y, int flags, void* param);
  * @param  userdate 	传给回调函数的参数
  */
void setMousecallback(const string& winname, MouseCallback onMouse, void* userdata=0);
```

```c++
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
```

## 6. 滑动条

```c++
/**
  * @brief	滑动条创建函数
  * @param 	trackbarname 	滑动条名称
  * @param 	winname 		窗口名称
  * @param  value 			滑动条初始值
  * @param  count 			滑动条最大值，最小值默认为0
  * @param  onChange 		回调函数指针，回调函数应有 void tracebar_callback(int value, void *userdata) 形式
  * @param  userdata  		返回数据
  */
int createTrackbar(const String& trackbarname, const String& winname,
                              int* value, int count,
                              TrackbarCallback onChange = 0,
                              void* userdata = 0);

/**
  * @brief	滑动条数值读取函数
  * @param 	trackbarname 	滑动条名称
  * @param 	winname 		窗口名称
  */
int getTrackbarPos(const String& trackbarname, const String& winname);
```

