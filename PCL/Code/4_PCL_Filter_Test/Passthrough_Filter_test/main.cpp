#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;

int main()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);			//待滤波点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);	//滤波后点云

	// 读入点云
	cout << "->正在读入点云..." << endl;
	if (pcl::io::loadPCDFile("test.pcd", *cloud) < 0)
	{
		PCL_ERROR("点云文件不存在！\n");
		system("pause");
		return -1;
	}
	cout << "\t\t<读入点云信息>\n" << *cloud << endl;

	// 直通滤波
	cout << "->正在进行直通滤波..." << endl;
	pcl::PassThrough<pcl::PointXYZ> pt;			// 创建滤波器对象
	pt.setInputCloud(cloud);					// 设置输入点云
	pt.setFilterFieldName("x");					// 设置滤波所需字段
	pt.setFilterLimits(-0.1, 1);				// 设置字段过滤范围
	// pt.setFilterLimitsNegative(true);		

	pt.filter(*cloud_filtered);					// 执行滤波，保存滤波后点云

	// 保存滤波后点云
	cout << "->正在保存点云...\n";
	if (cloud_filtered->empty())
	{
		PCL_ERROR("保存点云为空!\n");
		return -1;
	}
	else
	{
		pcl::io::savePCDFileASCII("filter.pcd", *cloud_filtered);
		cout << "\t\t<保存点云信息>\n" << *cloud_filtered << endl;
	}

	// 可视化

	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("滤波前后对比"));

	// 视图1
	int v1(0);
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1); //设置第一个视口在X轴、Y轴的最小值、最大值，取值在0-1之间
	viewer->setBackgroundColor(0, 0, 0, v1); //设置背景颜色，0-1，默认黑色（0，0，0）
	viewer->addText("befor_filtered", 10, 10, "v1_text", v1);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, "befor_filtered_cloud", v1);

	// 视图2
	int v2(0);
	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	viewer->setBackgroundColor(0.3, 0.3, 0.3, v2);
	viewer->addText("after_filtered", 10, 10, "v2_text", v2);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_filtered, "after_filtered_cloud", v2);

	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "befor_filtered_cloud", v1);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "befor_filtered_cloud", v1);

	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "after_filtered_cloud", v2);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "after_filtered_cloud", v2);

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
	}

	return 0;
}

