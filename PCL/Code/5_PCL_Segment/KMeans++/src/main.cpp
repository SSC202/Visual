#include "Kmeans.h"
#include <pcl/io/pcd_io.h>
#include <pcl/common/time.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace std;

int main()
{
	// 读取点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("test.pcd", *cloud) == -1)
	{
		PCL_ERROR("Could not read point cloud file! \n");
		return (-1);
	}
	cout << "Loaded " << cloud->size() << " points from point cloud" << endl;

	// K-means++ 聚类
	pcl::StopWatch time;
	int clusterNum = 7; // 聚类个数改为7
	int maxIter = 50;	// 最大迭代次数
	KMeans kmeans(clusterNum, maxIter);
	std::vector<pcl::Indices> cluster_indices;
	kmeans.extract(cloud, cluster_indices);
	cout << "Number of clusters: " << cluster_indices.size() << endl;
	cout << "Execution time: " << time.getTimeSeconds() << " seconds" << endl;

	// 聚类结果分类保存和可视化准备
	int begin = 1;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr all_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	// 预定义颜色表，确保颜色区分度
	std::vector<std::vector<uint8_t>> colors = {
		{255, 0, 0},   // 红色
		{0, 255, 0},   // 绿色
		{0, 0, 255},   // 蓝色
		{255, 255, 0}, // 黄色
		{255, 0, 255}, // 紫色
		{0, 255, 255}, // 青色
		{255, 128, 0}, // 橙色
		{128, 0, 255}, // 紫罗兰色
		{128, 255, 0}, // 黄绿色
		{0, 128, 255}  // 天蓝色
	};

	for (auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		// 获取每一个聚类点云团的点
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_kmeans(new pcl::PointCloud<pcl::PointXYZRGB>);

		// 使用预定义颜色，确保颜色区分度
		std::vector<uint8_t> color = colors[(begin - 1) % colors.size()];
		uint8_t R = color[0];
		uint8_t G = color[1];
		uint8_t B = color[2];

		for (auto pit = it->begin(); pit != it->end(); ++pit)
		{
			pcl::PointXYZRGB point_db;
			point_db.x = cloud->points[*pit].x;
			point_db.y = cloud->points[*pit].y;
			point_db.z = cloud->points[*pit].z;
			point_db.r = R;
			point_db.g = G;
			point_db.b = B;
			cloud_kmeans->points.push_back(point_db);
		}

		// 设置点云属性
		cloud_kmeans->width = cloud_kmeans->points.size();
		cloud_kmeans->height = 1;
		cloud_kmeans->is_dense = true;

		// 聚类结果分类保存
		pcl::io::savePCDFileBinary("kmeans_cluster_" + std::to_string(begin) + ".pcd", *cloud_kmeans);
		cout << "Cluster " << begin << " saved with " << cloud_kmeans->points.size() << " points" << endl;
		begin++;

		*all_cloud += *cloud_kmeans;
	}

	// 可视化

	// 创建可视化器
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("K-means++ Clustering Results (7 Clusters)"));
	viewer->setBackgroundColor(0.05, 0.05, 0.15); // 深蓝色背景

	// 左侧视口 - 原始点云
	int v1(0);
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer->setBackgroundColor(0.1, 0.1, 0.2, v1);

	// 添加原始点云（白色）
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_white(cloud, 255, 255, 255);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_white, "original_cloud", v1);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud", v1);

	// 添加文本说明
	viewer->addText("Original Point Cloud", 10, 20, 16, 1, 1, 1, "original_text", v1);
	std::string original_count = "Points: " + std::to_string(cloud->size());
	viewer->addText(original_count, 10, 40, 14, 1, 1, 1, "original_count", v1);

	// 右侧视口 - 聚类结果
	int v2(0);
	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	viewer->setBackgroundColor(0.1, 0.15, 0.1, v2);

	// 添加聚类结果点云
	viewer->addPointCloud<pcl::PointXYZRGB>(all_cloud, "clustered_cloud", v2);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "clustered_cloud", v2);

	// 添加文本说明
	viewer->addText("K-means++ Clustering Results", 10, 20, 16, 1, 1, 1, "clustered_text", v2);
	std::string cluster_count = "Clusters: " + std::to_string(cluster_indices.size());
	viewer->addText(cluster_count, 10, 40, 14, 1, 1, 1, "cluster_count", v2);

	// 添加参数信息
	std::string params_info = "Clusters: " + std::to_string(clusterNum) + ", MaxIter: " + std::to_string(maxIter);
	viewer->addText(params_info, 10, 60, 14, 1, 1, 1, "params_info", v2);

	std::string time_info = "Time: " + std::to_string(time.getTimeSeconds()) + "s";
	viewer->addText(time_info, 10, 80, 14, 1, 1, 1, "time_info", v2);

	// 公共设置

	// 添加标题
	viewer->addText("K-means++ Clustering (7 Clusters)", 250, 20, 18, 1, 1, 1, "title");

	// 添加坐标系
	viewer->addCoordinateSystem(1.0, "axis_v1", v1);
	viewer->addCoordinateSystem(1.0, "axis_v2", v2);

	// 设置相机参数
	viewer->initCameraParameters();
	viewer->setCameraPosition(0, 0, 10, 0, 0, 0, 0, 1, 0);

	cout << "\n=== Visualization Started ===" << endl;
	cout << "Left: Original point cloud (White)" << endl;
	cout << "Right: K-means++ clustering results (7 colored clusters)" << endl;
	cout << "Parameters: Clusters=" << clusterNum << ", MaxIterations=" << maxIter << endl;
	cout << "Press 'q' to exit" << endl;
	cout << "Use mouse to rotate and scroll to zoom" << endl;

	// 主循环
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}

	return 0;
}