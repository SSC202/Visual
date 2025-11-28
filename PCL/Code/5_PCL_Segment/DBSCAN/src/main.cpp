#include <iostream>
#include <string>
#include "dbscan.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;

int main()
{
	// 读取数据
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	if (pcl::io::loadPCDFile<pcl::PointXYZ>("test.pcd", *cloud) < 0)
	{
		PCL_ERROR("点云读取失败！！！ \n");
		return -1;
	}
	cout << "从点云数据中读取：" << cloud->points.size() << "个点" << endl;

	// 密度聚类
	pcl::StopWatch time;
	vector<pcl::Indices> cluster_indices;
	dbscan(*cloud, cluster_indices, 1, 50); // 2表示聚类的领域距离为2米，50表示聚类的最小点数。

	cout << "密度聚类的个数为：" << cluster_indices.size() << endl;
	cout << "代码运行时间:" << time.getTimeSeconds() << "秒" << endl;

	// 聚类结果分类保存
	int begin = 1;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr dbscan_all_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	for (vector<pcl::Indices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		// 获取每一个聚类点云团的点
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_dbscan(new pcl::PointCloud<pcl::PointXYZRGB>);
		// 同一点云团赋上同一种颜色
		uint8_t R = rand() % (256) + 0;
		uint8_t G = rand() % (256) + 0;
		uint8_t B = rand() % (256) + 0;

		for (auto pit = it->begin(); pit != it->end(); ++pit)
		{
			pcl::PointXYZRGB point_db;
			point_db.x = cloud->points[*pit].x;
			point_db.y = cloud->points[*pit].y;
			point_db.z = cloud->points[*pit].z;
			point_db.r = R;
			point_db.g = G;
			point_db.b = B;
			cloud_dbscan->points.push_back(point_db);
		}
		// 聚类结果分类保存
		stringstream ss;
		ss << "dbscan_cluster_" << begin << ".pcd";
		pcl::PCDWriter writer;
		writer.write<pcl::PointXYZRGB>(ss.str(), *cloud_dbscan, true);
		begin++;

		*dbscan_all_cloud += *cloud_dbscan;
	}

	// 聚类结果可视化
	pcl::visualization::CloudViewer viewer("DBSCAN cloud viewer.");
	viewer.showCloud(dbscan_all_cloud);
	while (!viewer.wasStopped())
	{

	}
	return 0;
}

