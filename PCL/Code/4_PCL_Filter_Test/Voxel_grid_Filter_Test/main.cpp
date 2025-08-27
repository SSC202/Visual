#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;

int main()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);				//待滤波点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);		//滤波后点云

	// 读入点云数据
	cout << "->正在读入点云..." << endl;
	pcl::PCDReader reader;
	reader.read("test.pcd", *cloud);
	cout << "\t\t<读入点云信息>\n" << *cloud << endl;

	// 体素滤波器点云下采样
	cout << "->正在体素下采样..." << endl;
	pcl::VoxelGrid<pcl::PointXYZ> vg;				// 创建滤波器对象
	vg.setInputCloud(cloud);						// 设置待滤波点云
	vg.setLeafSize(0.05f, 0.05f, 0.05f);			// 设置体素大小
	vg.filter(*cloud_filtered);						// 执行滤波，保存滤波结果于cloud_filtered

	// 保存下采样点云
	cout << "->正在保存下采样点云..." << endl;
	pcl::PCDWriter writer;
	writer.write("sub.pcd", *cloud_filtered, true);
	cout << "\t\t<保存点云信息>\n" << *cloud_filtered << endl;

	// 可视化

	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("滤波前后对比"));

	// 视图1
	int v1(0);
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1); 
	viewer->setBackgroundColor(0, 0, 0, v1); 
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

