#include <pcl/point_cloud.h>					// 点云类型
#include <pcl/kdtree/kdtree_flann.h>			// KDtree相关定义
#include <pcl/visualization/cloud_viewer.h>		// 可视化相关定义

#include <iostream>
#include <vector>
#include <ctime>

#include <thread>
#include <chrono>

using namespace std;

int main(int argc, char** argv)
{

	// 使用系统时间做随机数种子
	srand(time(NULL));

	// 创建一个PointXYZ类型点云指针
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	// 初始化点云数据
	cloud->width = 1000;// 宽为1000
	cloud->height = 1;//高为1，说明为无序点云
	cloud->points.resize(cloud->width * cloud->height);

	// 使用随机数填充数据
	for (size_t i = 0; i < cloud->size(); ++i)
	{
		cloud->points[i].x = 1024.0f * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].y = 1024.0f * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].z = 1024.0f * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].b = 0;
		cloud->points[i].g = 255;
		cloud->points[i].r = 0;
	}

	// 创建 k-d tree
	pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;

	// 设置点云输入,将在cloud中搜索
	kdtree.setInputCloud(cloud);

	// 设置被搜索点,用随机数填充
	pcl::PointXYZRGB searchPoint;
	searchPoint.x = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.y = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.z = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.b = 0;
	searchPoint.g = 0;
	searchPoint.r = 255;

	// 开始 KNN 搜索,K设置为10
	int K = 10;

	// 基于半径的邻域搜索
	float radius = 256.0f * rand() / (RAND_MAX + 1.0f);

	// 存储搜索结果
	vector<int> pointIdxRadiusSearch;
	vector<float> pointRadiusSquaredDistance;

	// 存储搜索结果
	vector<int> pointIdxNKNSearch(K);			// 保存下标
	vector<float> pointNKNSquaredDistance(K);	// 保存距离的平方

	// KNN
	cout << "K nearest neighbor search at (" << searchPoint.x
		<< " " << searchPoint.y
		<< " " << searchPoint.z
		<< ") with K = " << K << endl;

	if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
	{
		for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
		{
			cout << "    " << cloud->points[pointIdxNKNSearch[i]].x
				<< " " << cloud->points[pointIdxNKNSearch[i]].y
				<< " " << cloud->points[pointIdxNKNSearch[i]].z
				<< "( squared distance: " << pointNKNSquaredDistance[i] << " )" << endl;
			// 查询点邻域内的点着色
			cloud->points[pointIdxNKNSearch[i]].r = 0;
			cloud->points[pointIdxNKNSearch[i]].g = 0;
			cloud->points[pointIdxNKNSearch[i]].b = 255;
		}
	}

	// Radius Search
	cout << "Neighbors within radius search at (" << searchPoint.x
		<< " " << searchPoint.y
		<< " " << searchPoint.z
		<< ") with radius=" << radius << endl;

	// 标记搜索结果
	if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
	{
		for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
		{
			cout << "    " << cloud->points[pointIdxRadiusSearch[i]].x
				<< " " << cloud->points[pointIdxRadiusSearch[i]].x
				<< " " << cloud->points[pointIdxRadiusSearch[i]].z
				<< "( squared distance: " << pointRadiusSquaredDistance[i] << " )" << endl;
			// 查询点邻域内的点着色
			cloud->points[pointIdxRadiusSearch[i]].r = 255;
			cloud->points[pointIdxRadiusSearch[i]].g = 0;
			cloud->points[pointIdxRadiusSearch[i]].b = 255;
		}
	}

	// 可视化
	pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	viewer.setBackgroundColor(0.0, 0.0, 0.0);
	viewer.addPointCloud<pcl::PointXYZRGB>(cloud, "cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");

	while (!viewer.wasStopped()) {
		viewer.spinOnce();
		std::this_thread::sleep_for(std::chrono::microseconds(100));
	}

	return 0;
}
