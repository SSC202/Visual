#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/visualization/cloud_viewer.h>

#include <X11/Xlib.h>

using namespace std;

int main()
{
	XInitThreads();

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);			//原始点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected(new pcl::PointCloud<pcl::PointXYZ>);		//投影点云

	///读入点云数据
	cout << "->正在读入点云..." << endl;
	pcl::PCDReader reader;
	reader.read("test.pcd", *cloud);
	cout << "\t\t<读入点云信息>\n" << *cloud << endl;

	///参数化模型投影
	cout << "->正在平面模型投影..." << endl;
	//创建 x+y+z=0 平面
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	coefficients->values.resize(4);	//设置模型系数的大小
	coefficients->values[0] = 1.0;	//x系数
	coefficients->values[1] = 1.0;	//y系数
	coefficients->values[2] = 1.0;	//z系数
	coefficients->values[3] = 0.0;	//常数项
	//投影滤波
	pcl::ProjectInliers<pcl::PointXYZ> proj;//创建投影滤波器对象
	proj.setModelType(pcl::SACMODEL_PLANE);	//设置对象对应的投影模型
	proj.setInputCloud(cloud);				//设置输入点云
	proj.setModelCoefficients(coefficients);//设置模型对应的系数
	proj.filter(*cloud_projected);			//执行投影滤波，存储结果于cloud_projected

	///保存滤波点云
	cout << "->正在保存投影点云..." << endl;
	pcl::PCDWriter writer;
	writer.write("proj_PLANE.pcd", *cloud_projected, true);
	cout << "\t\t<保存点云信息>\n" << *cloud_projected << endl;

	//================================= 滤波前后对比可视化 ================================= ↓

	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("滤波前后对比"));

	/*-----视口1-----*/
	int v1(0);
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1); //设置第一个视口在X轴、Y轴的最小值、最大值，取值在0-1之间
	viewer->setBackgroundColor(0, 0, 0, v1); //设置背景颜色，0-1，默认黑色（0，0，0）
	viewer->addText("befor_filtered", 10, 10, "v1_text", v1);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, "befor_filtered_cloud", v1);

	/*-----视口2-----*/
	int v2(0);
	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	viewer->setBackgroundColor(0.3, 0.3, 0.3, v2);
	viewer->addText("after_filtered", 10, 10, "v2_text", v2);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_projected, "after_filtered_cloud", v2);

	/*-----设置相关属性-----*/
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "befor_filtered_cloud", v1);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "befor_filtered_cloud", v1);

	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "after_filtered_cloud", v2);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "after_filtered_cloud", v2);

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
	}

	//================================= 滤波前后对比可视化 ================================= ↑

	return 0;
}

