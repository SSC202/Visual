# OpenCV 7_图像直方图

图像直方图是图像处理中非常重要的像素统计结果，图像直方图不再表征任何的图像纹理信息，而是对图像像素的统计。由于同一物体无论是旋转还是平移在图像中都具有相同的灰度值，因此直方图具有平移不变性、放缩不变性等优点，因此可以用来查看图像整体的变化形式，例如图像是否过暗、图像像素灰度值主要集中在哪些范围等，在特定的条件下也可以利用图像直方图进行图像的识别，例如对数字的识别。

图像直方图简单来说就是统计图像中每个灰度值的个数，之后将图像灰度值作为横轴，以灰度值个数或者灰度值所占比率作为纵轴绘制的统计图。通过直方图可以看出图像中哪些灰度值数目较多，哪些较少，可以通过一定的方法将灰度值较为集中的区域映射到较为稀疏的区域，从而使得图像在像素灰度值上分布更加符合期望状态。通常情况下，像素灰度值代表亮暗程度，因此通过图像直方图可以分析图像亮暗对比度，并调整图像的亮暗程度。

## 1. 直方图的绘制
```c++
void cv::calcHist(const Mat * images,
                        int  nimages,
                        const int * channels,
                        InputArray mask,
                        OutputArray hist,
                        int  dims,
                        const int * histSize,
                        const float ** ranges,
                        bool  uniform = true,
                        bool  accumulate = false 
                 );
```
>images：待统计直方图的图像数组，数组中所有的图像应具有相同的尺寸和数据类型，并且数据类型只能是CV_8U、CV_16U和CV_32F三种中的一种，但是不同图像的通道数可以不同。
>
>nimages：输入的图像数量
>
>channels：需要统计的通道索引数组，第一个图像的通道索引从0到`images[0].
channels()-1`，第二个图像通道索引从`images[0].channels()`到`images[0].channels()+images[1].channels()-1`，以此类推。
>
>mask：可选的操作掩码，如果是空矩阵则表示图像中所有位置的像素都计入直方图中，如果矩阵不为空，则必须与输入图像尺寸相同且数据类型为CV_8U。
>
>hist：输出的统计直方图结果，是一个dims维度的数组。
>
>dims：需要计算直方图的维度，必须是整数，并且不能大于CV_MAX_DIMS。
>
>histSize：存放每个维度直方图的数组的尺寸。
>
>ranges：每个图像通道中灰度值的取值范围。
>
>uniform：直方图是否均匀的标志符，默认状态下为均匀（true）。
>
>accumulate：是否累积统计直方图的标志，如果累积（true），则统计新图像的直方图时之前图像的统计结果不会被清除，该同能主要用于统计多个图像整体的直方图。

```c++
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat img = imread("picture.jpg");
	if (img.empty())
	{
		cout << "Fail to open!" << endl;
		return -1;
	}
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	//直方图数据设置
	Mat hist;
	const int channels[1] = { 0 };           //通道索引
	float inRanges[2] = { 0,255 };
	const float* ranges[1] = { inRanges };   //像素灰度值范围
	const int dims[1] = { 256 };             //直方图维度（像素灰度值最大值）
	//直方图生成
	calcHist(&img, 1, channels, Mat(), hist, 1, dims, ranges);

	//直方图绘制
	int hist_w = 512;
	int hist_h = 400;
	int width = 2;
	Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
	for (int i = 1; i <= hist.rows; ++i)
	{
		rectangle(histImage, Point(width * (i - 1), hist_h - 1), Point(width * i - 1, hist_h - cvRound(hist.at<float>(i - 1) / 20)), Scalar(255, 255, 255), -1);
	}
	namedWindow("histImage", WINDOW_AUTOSIZE);
	imshow("histImage", histImage);
	imshow("gray", gray);
	waitKey(0);
	return 0;
}
```
## 2. 直方图归一化
由于绘制直方图的图像高度小于某些灰度值统计的数目，因此我们在绘制直方图时将所有的数据都缩小为原来的二十分之一之后再进行绘制，目的就是为了能够将直方图完整的绘制在图像中。如果换一张图像的直方图统计结果或者将直方图绘制到一个尺寸更小的图像中时，可能需要将统计数据缩小为原来的三十分之一、五十分之一甚至更低。数据缩小比例与统计结果、将要绘制直方图图像的尺寸相关，因此每次绘制时都需要计算数据缩小的比例。另外，由于像素灰度值统计的数目与图像的尺寸具有直接关系，如果以灰度值数目作为最终统计结果，那么一张图像经过尺寸放缩后的两张图像的直方图将会有巨大的差异，然而直方图可以用来表示图像的明亮程度，从理论上讲通过缩放的两张图像将具有大致相似的直方图分布特性，因此用灰度值的数目作为统计结果具有一定的局限性。

图像的像素灰度值统计结果主要目的之一就是查看某个灰度值在所有像素中所占的比例，因此可以用每个灰度值像素的数目占一幅图像中所有像素数目的比例来表示某个灰度值数目的多少，即将统计结果再除以图像中像素个数。这种方式可以保证每个灰度值的统计结果都是0到100%之间的数据，实现统计结果的归一化。为了更直观的绘制图像直方图，常需要将比例扩大一定的倍数后再绘制图像。另一种常用的归一化方式是寻找统计结果中最大数值，把所有结果除以这个最大的数值，以实现将所有数据都缩放到0到1之间。

```c++
void cv::normalize(InputArray src,
                   InputOutputArray dst,
                   double  alpha = 1,
                   double   beta = 0,
                   int  norm_type = NORM_L2,
                   int  dtype = -1,
                   InputArray mask = noArray()
                  );
```
>src：输入数组矩阵。
>
>dst：输入与src相同大小的数组矩阵。
>
>alpha：在范围归一化的情况下，归一化到下限边界的标准值
>
>beta：范围归一化时的上限范围，它不用于标准规范化。
>
>norm_type：归一化过程中数据范数种类标志.
>
>dtype：输出数据类型选择标志，如果为负数，则输出数据与src拥有相同的类型，否则与src具有相同的通道数和数据类型。
>
>mask：掩码矩阵。

选择NORM_L1标志，输出结果为每个灰度值所占的比例；选择NORM_INF参数，输出结果为除以数据中最大值，将所有的数据归一化到0到1之间。

>morm_type|值|含义
>-|-|-
>NORM_INF|1|无穷范数，向量最大值（最大值归一化）
>NORM_L1|2|L1范数，绝对值之和（绝对值求和归一化）
>NORM_L2|3|L2范数，平方和之根（模长归一化）
>NORM_L2SQR|4|L2范数平方
>NORM_MINMAX||偏移归一化

```c++
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat img = imread("picture.jpg");
	if (img.empty())
	{
		cout << "Fail to open!" << endl;
		return -1;
	}

	Mat src;
	cvtColor(img, src, COLOR_BGR2GRAY);

	//直方图数据配置
	Mat hist;
	const int channels[1] = { 0 };
	float inranges[2] = { 0,255 };
	const float* ranges[1] = { inranges };
	const int dims[1] = {256};

	calcHist(&src, 1, channels, Mat(), hist, 1, dims, ranges);

	int hist_w = 512;
	int hist_h = 400;
	int width = 2;

	Mat Img_image_1 = Mat::zeros(hist_h, hist_w, CV_8UC3);
	Mat Img_image_2 = Mat::zeros(hist_h, hist_w, CV_8UC3);
	Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
	Mat hist_1, hist_2;
	normalize(hist, hist_1, 1, 0, NORM_L2, -1, Mat());
	normalize(hist, hist_2, 1, 0, NORM_L1, -1, Mat());

	for (int i = 1; i <= hist.rows; ++i)
	{
		rectangle(Img_image_1, Point(width * (i - 1), hist_h - 1),Point(width * i - 1, hist_h - cvRound(hist_h * hist_1.at<float>(i - 1)) - 1),Scalar(255, 255, 255), -1);
	}

	for (int i = 1; i <= hist.rows; ++i)
	{
		rectangle(Img_image_2, Point(width * (i - 1), hist_h - 1), Point(width * i - 1, hist_h - cvRound(hist_h * hist_2.at<float>(i - 1)) - 1), Scalar(255, 255, 255), -1);
	}

	for (int i = 1; i <= hist.rows; ++i)
	{
		rectangle(histImage, Point(width * (i - 1), hist_h - 1), Point(width * i - 1, hist_h - cvRound(hist_h * hist.at<float>(i - 1)) - 1), Scalar(255, 255, 255), -1);
	}
	imshow("IMG_HIST", histImage);
	imshow("histImage_L2", Img_image_1);
	imshow("histImage_L1", Img_image_2);
	imshow("img", src);
	waitKey();

	return 0;
}
```
## 3. 直方图比较
从一定程度上来讲，虽然两张图像的直方图分布相似不代表两张图像相似，但是两张图像相似则两张图像的直方图分布一定相似。例如通过插值对图像进行放缩后图像的直方图虽然不会与之前完全一致，但是两者一定具有很高的相似性，因而可以通过比较两张图像的直方图分布相似性对图像进行初步的筛选与识别。
```c++
double cv::compareHist(InputArray H1,
                       InputArray H2,
                       int  method
                      );
```
>H1：第一张图像直方图。
>
>H2：第二张图像直方图，与H1具有相同的尺寸
>
>method：比较方法标志

该函数前两个参数为需要比较相似性的图像直方图，由于不同尺寸的图像中像素数目可能不相同，为了能够得到两个直方图图像正确的相识性，需要输入同一种方式归一化后的图像直方图，并且要求两个图像直方图具有相同的尺寸。

>method|值|作用
>-|-|-
>HISTCMP_CORREL|0|相关法
>HISTCMP_CHISQR|1|卡方法
>HISTCMP_INTERSECT|2|直方图相交法
>HISTCMP_BHATTACHARYYA|3|巴氏距离法
>HISTCMP_CHISQR_ALT|4|替代卡方法
>HISTCMP_KL_DIV|5|相对熵法

- 相关法

在该方法中如果两个图像直方图完全一致，则计算数值为1；如果两个图像直方图完全不相关，则计算值为0。
$$
d(H_1,H_2) = \frac{\Sigma_I(H_1(I)-\overline{H_1})(H_2(I)-\overline{H_2})}{\sqrt{\Sigma_I(H_1(I)-\overline{H_1})^2\Sigma_I(H_2(I)-\overline{H_2})^2}}
\\
\overline{H_k} = \frac{1}{N}\Sigma_JH_k(J)
$$
N是直方图的灰度值个数。

- 卡方法
$$
d(H_1,H_2)=\Sigma_I\frac{H_1(I)-H_2(I)^2}{H_1}
$$

- 直方图相交法

该方法不会将计算结果归一化，因此即使是两个完全一致的图像直方图，来自于不同图像也会有不同的数值，但是其遵循与同一个图像直方图比较时，数值越大相似性越高，数值越小相似性越低。

$$
d(H_1,H_2)=\Sigma_Imin(H_1(I),H_2(I))
$$

- 巴氏距离法
$$
d(H_1,H_2)=\sqrt{1-\frac{1}{\sqrt{H_1H_2N^2}}\Sigma_I\sqrt{H_1(I)H_2(I)}}
$$

- 替代卡方法
$$
d(H_1,H_2)=2\Sigma_I\frac{{(H_1(I)-H_2(I))}^2}{H_1(I)+H_2(I)}
$$
- 相对熵法
$$
d(H_1,H_2)=\Sigma_IH_1(I)log(\frac{H_1(I)}{H_2(I)})
$$

```c++
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
#define WIDTH 2
#define HIST_W 512
#define HIST_H 400
void drawHist(Mat& hist, Mat& hist_dst, int width, int hist_h, int hist_w, int type);

int main()
{
	Mat img_1 = imread("picture_1.jpg");
	Mat img_2 = imread("picture_2.jpg");

	if (img_1.empty() || img_2.empty())
	{
		cout << "Fail to open!" << endl;
		return -1;
	}
	cvtColor(img_1, img_1, COLOR_BGR2GRAY);
	cvtColor(img_2, img_2, COLOR_BGR2GRAY);

	Mat hist_1,hist_2,hist_dst_1,hist_dst_2;
	const int channel[1] = { 0 };
	float inrange[2] = { 0,255 };
	const float* range[1] = {inrange};
	const int dims[1] = { 256 };

	calcHist(&img_1, 1, channel, Mat(), hist_1, 1, dims, range);
	calcHist(&img_2, 1, channel, Mat(), hist_2, 1, dims, range);

	drawHist(hist_1, hist_dst_1, WIDTH, HIST_H, HIST_W, NORM_L1);
	drawHist(hist_2, hist_dst_2, WIDTH, HIST_H, HIST_W, NORM_L1);
	system("cls");
	double alpha = compareHist(hist_dst_1, hist_dst_2, HISTCMP_CORREL);
	cout << "alpha = " << alpha << endl;

//	imshow("img_1", img_1);
//	imshow("img_2", img_2);
	waitKey();

	return 0;
}

void drawHist(Mat &hist,Mat &hist_dst,int width,int hist_h,int hist_w,int type)
{
	Mat Img_image = Mat::zeros(hist_h, hist_w, CV_8UC3);
	normalize(hist, hist_dst, 1, 0, type, -1, Mat());
	for (int i = 1; i <= hist.rows; ++i)
	{
		rectangle(Img_image, Point(width * (i - 1), hist_h - 1), Point(width * i - 1, hist_h - cvRound(hist_h * hist_dst.at<float>(i - 1)) - 1), Scalar(255, 255, 255), -1);
	}
}
```

## 4. 直方图均衡化

如果一个图像的直方图都集中在一个区域，则整体图像的对比度比较小，不便于图像中纹理的识别。如果通过映射关系，将图像中灰度值的范围扩大，增加原来两个灰度值之间的差值，就可以提高图像的对比度，进而将图像中的纹理突出显现出来，这个过程称为图像直方图均衡化。
```c++
void cv::equalizeHist(InputArray src,OutputArray dst);
```
>src：需要直方图均衡化的CV_8UC1图像。
>
>dst：直方图均衡化后的输出图像，与src具有相同尺寸和数据类型。

```c++
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#define WIDTH 2
#define HIST_W 512
#define HIST_H 400
void createHist(Mat& img, Mat& hist);
void drawHist(Mat& hist, Mat& hist_dst, int type,string str, int width = WIDTH, int hist_h = HIST_H, int hist_w = HIST_W);

int main()
{
	Mat img = imread("picture.jpg");
	if (img.empty())
	{
		cout << "Fail to open!" << endl;
		return -1;
	}

	cvtColor(img, img, COLOR_BGR2GRAY);
	Mat hist,hist_dst,eimg,ehist,ehist_dst;
	equalizeHist(img, eimg);

	createHist(img, hist);
	drawHist(hist, hist_dst, NORM_L1,"hist");
	createHist(eimg, ehist);
	drawHist(ehist, ehist_dst, NORM_L1,"ehist");

	double alpha = compareHist(ehist_dst, hist_dst, HISTCMP_CORREL);
	system("cls");
	cout << "alpha = " << alpha << endl;
	
	imshow("eimg", eimg);

	waitKey();

	return 0;
}

void createHist(Mat& img,Mat &hist)
{
	const int channel[1] = { 0 };
	float inrange[2] = { 0,255 };
	const float* range[1] = { inrange };
	const int dims[1] = { 256 };
	calcHist(&img, 1, channel, Mat(), hist, 1, dims, range);
}

void drawHist(Mat& hist, Mat& hist_dst, int type,string str, int width, int hist_h, int hist_w)
{
	Mat Img_image = Mat::zeros(hist_h, hist_w, CV_8UC3);
	normalize(hist, hist_dst, 1, 0, type, -1, Mat());
	for (int i = 1; i <= hist.rows; ++i)
	{
		rectangle(Img_image, Point(width * (i - 1), hist_h - 1), Point(width * i - 1, hist_h - cvRound(hist_h * hist_dst.at<float>(i - 1)) - 1), Scalar(255, 255, 255), -1);
	}
	imshow(str, Img_image);
}
```

## 5. 直方图匹配
直方图均衡化函数可以自动的改变图像直方图的分布形式，这种方式极大的简化了直方图均衡化过程中需要的操作步骤，但是该函数不能指定均衡化后的直方图分布形式。在某些特定的条件下需要将直方图映射成指定的分布形式，这种将直方图映射成指定分布形式的算法称为直方图匹配或者直方图规定化。直方图匹配与直方图均衡化相似，都是对图像的直方图分布形式进行改变，只是直方图均衡化后的图像直方图是均匀分布的，而直方图匹配后的直方图可以随意指定，即在执行直方图匹配操作时，首先要知道变换后的灰度直方图分布形式，进而确定变换函数。直方图匹配操作能够有目的的增强某个灰度区间。

由于不同图像间像素数目可能不同，为了使两个图像直方图能够匹配，需要使用概率形式去表示每个灰度值在图像像素中所占的比例。理想状态下，经过图像直方图匹配操作后图像直方图分布形式应与目标分布一致，因此**两者之间的累积概率分布也一致。累积概率为小于等于某一灰度值的像素数目占所有像素中的比例。**

寻找灰度值匹配的过程是直方图匹配算法的关键，在代码实现中可以**通过构建原直方图累积概率与目标直方图累积概率之间的差值表，寻找原直方图中灰度值n的累积概率与目标直方图中所有灰度值累积概率差值的最小值，这个最小值对应的灰度值r就是n匹配后的灰度值。**

```c++
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<math.h>
using namespace std;
using namespace cv;

#define WIDTH 2
#define HIST_W 512
#define HIST_H 400
void createHist(Mat& img, Mat& hist);
void drawHist(Mat& hist, Mat& hist_dst, int type, string str, int width = WIDTH, int hist_h = HIST_H, int hist_w = HIST_W);

int main()
{
	Mat img = imread("picture.jpg");
	Mat f_img = imread("format.jpg");

	if (img.empty() || f_img.empty())
	{
		cout << "Fail to open!" << endl;
		return -1;
	}
	cvtColor(img, img, COLOR_BGR2GRAY);
	cvtColor(f_img, f_img, COLOR_BGR2GRAY);
	resize(f_img, f_img, Size(img.cols, img.rows));
	Mat hist, hist_f;
	createHist(img, hist);
	createHist(f_img, hist_f);
	drawHist(hist, hist, NORM_INF, "hist");
	drawHist(hist_f, hist_f, NORM_INF, "hist_f");

	//构建累积概率矩阵
	float hist1_cdf[256] = { hist.at<float>(0) };
	float hist2_cdf[256] = { hist_f.at<float>(0) };
	for (int i = 1; i < 256; ++i) {
		hist1_cdf[i] = hist1_cdf[i - 1] + hist.at<float>(i);
		hist2_cdf[i] = hist2_cdf[i - 1] + hist_f.at<float>(i);
	}
	//构建累积概率误差矩阵
	float diff_cdf[256][256];
	for (int i = 0; i < 256; ++i) {
		for (int j = 0; j < 256; ++j) {
			diff_cdf[i][j] = fabs(hist1_cdf[i] - hist2_cdf[j]);
		}
	}
	//生成LUT映射表
	Mat lut(1, 256, CV_8U);
	for (int i = 0; i < 256; ++i) {
		//查找源灰度级为i的映射灰度
		//和i的累积概率差值的最小的规定灰度
		float min = diff_cdf[i][0];
		int index = 0;
		//寻找累积概率误差矩阵中每一行中的最小值
		for (int j = 1; j < 256; ++j) {
			if (min > diff_cdf[i][j]) {
				min = diff_cdf[i][j];
				index = j;
			}
		}
		lut.at<uchar>(i) = (uchar)index;
	}
	Mat r_hist, res;
	LUT(img, lut, res);
	imshow("img", img);
	imshow("format", f_img);
	imshow("result", res);
	createHist(res, r_hist);
	drawHist(r_hist,r_hist, NORM_L1, "r_hist");
	waitKey();



	return 0;
}

void createHist(Mat& img, Mat& hist)
{
	const int channel[1] = { 0 };
	float inrange[2] = { 0,255 };
	const float* range[1] = { inrange };
	const int dims[1] = { 256 };
	calcHist(&img, 1, channel, Mat(), hist, 1, dims, range);
}

void drawHist(Mat& hist, Mat& hist_dst, int type, string str, int width, int hist_h, int hist_w)
{
	Mat Img_image = Mat::zeros(hist_h, hist_w, CV_8UC3);
	normalize(hist, hist_dst, 1, 0, type, -1, Mat());
	for (int i = 1; i <= hist.rows; ++i)
	{
		rectangle(Img_image, Point(width * (i - 1), hist_h - 1), Point(width * i - 1, hist_h - cvRound(hist_h * hist_dst.at<float>(i - 1)) - 1), Scalar(255, 255, 255), -1);
	}
	imshow(str, Img_image);
}
```

## 6. 模板匹配

直方图不能直接反应图像的纹理，因此如果两张不同模板图像具有相同的直方图分布特性，那么在同一张图中对这两张模板图像的直方图进行反向投影，最终结果将不具有参考意义。因此，我们在图像中寻找模板图像时，可以直接通过比较图像像素的形式来搜索是否存在相同的内容，这种通过比较像素灰度值来寻找相同内容的方法叫做图像的模板匹配。

```c++
void cv::matchTemplate(InputArray image,
                       InputArray templ,
                       OutputArray result,
                       int  method,
                       InputArray mask = noArray()
                      );
```
>image：待模板匹配的原图像，图像数据类型为CV_8U和CV_32F两者中的一个。
>
>templ：模板图像，需要与image具有相同的数据类型，但是尺寸不能大于image。
>
>result：模板匹配结果输出图像，图像数据类型为CV_32F。如果image的尺寸为W×H，模板图像尺寸为w×h，则输出图像的尺寸为（W-w+1）×（H-h+1）。
>
>method：模板匹配方法标志。
>
>mask：匹配模板的掩码，必须与模板图像具有相同的数据类型和尺寸，默认情况下不设置，目前仅支持在TM_SQDIFF和TM_CCORR_NORMED这两种匹配方法时使用。

>method|值|含义
>-|-|-
>TM_SQDIF|0|平方差匹配法
>TM_SQDIFF_NORMED|1|归一化平方差匹配法
>TM_CCORR|2|相关匹配法
>TM_CCORR_NORMED|3|归一化相关匹配法
>TM_CCOEFF|4|系数匹配法
>TM_CCOEFF_NORMED|5|归一化相关系数匹配法

- TM_SQDIFF

当模板与滑动窗口完全匹配时计算数值为0，两者匹配度越低计算数值越大。
$$
R(x,y)= \Sigma(T(x^,,y^,)-I(x+x^,,y+y^,))^2
$$
其中T表示模板图像，I表示原图像。

- TM_SQDIFF_NORMED
$$
R(x,y)= \frac{\Sigma(T(x^,,y^,)-I(x+x^,,y+y^,))^2}{\sqrt{\Sigma_{x^,,y^,}T(x^,,y^,)^2*\Sigma_{x^,,y^,}I(x+x^,,y+y^,)^2}}
$$

- TM_CCORR

数值越大表示匹配效果越好，0表示最坏的匹配结果。
$$
R(x,y)= \Sigma(T(x^,,y^,)*I(x+x^,,y+y^,))
$$

- TM_CCORR_NORMED
$$
R(x,y)= \frac{\Sigma(T(x^,,y^,)*I(x+x^,,y+y^,))}{\sqrt{\Sigma_{x^,,y^,}T(x^,,y^,)^2*\Sigma_{x^,,y^,}I(x+x^,,y+y^,)^2}}
$$

- TM_CCOEFF

这种方法采用相关匹配方法对模板减去均值的结果和原图像减去均值的结果进行匹配，这种方法可以很好的解决模板图像和原图像之间由于亮度不同而产生的影响。该方法中模板与滑动窗口匹配度越高计算数值越大，匹配度越低计算数值越小，并且该方法计算结果可以为负数。

$$
R(x,y)= \Sigma(T^,(x^,,y^,)*I^,(x+x^,,y+y^,))\\
\begin{cases}
\ T^,(x^,,y^,)=T(x^,,y^,)-\frac{1}{w*h}\Sigma_{x^{,,},y^{,,}}T(x^{,,},y^{,,}) \\
\ I^,(x+x^,,y+y^,)=I(x+x^,,y+y^,)-\frac{1}{w*h}\Sigma_{x^{,,},y^{,,}}I(x+x^{,,},y+y^{,,})
\end{cases}
$$

- TM_CCOEFF_NORMED

$$
R(x,y)= \frac{\Sigma(T^,(x^,,y^,)*I^,(x+x^,,y+y^,))}{\sqrt{\Sigma_{x^,,y^,}T(x^,,y^,)^2*\Sigma_{x^,,y^,}I(x+x^,,y+y^,)^2}}
$$

由于matchTemplate()函数的输出结果是存有相关性系数的矩阵，**因此需要通过minMaxLoc()函数去寻找输入矩阵中的最大值或者最小值，进而确定模板匹配的结果。**

通过寻找输出矩阵的最大值或者最小值得到的只是一个像素点，需要以该像素点为矩形区域的左上角，绘制与模板图像同尺寸的矩形框，标记出最终匹配的结果。
