#include "Kmeans.h"
#include <pcl/io/pcd_io.h>
#include <pcl/common/angles.h>
#include <pcl/common/time.h>

using namespace std;

int main()
{
	// 加载点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	// 构造球体
	float radius = 2;
	for (float r = 0; r < radius; r += 0.1)
	{
		for (float angle1 = 0.0; angle1 <= 180.0; angle1 += 5.0)
		{
			for (float angle2 = 0.0; angle2 <= 360.0; angle2 += 5.0)
			{
				pcl::PointXYZ basic_point;

				basic_point.x = radius * sinf(pcl::deg2rad(angle1)) * cosf(pcl::deg2rad(angle2));
				basic_point.y = radius * sinf(pcl::deg2rad(angle1)) * sinf(pcl::deg2rad(angle2));
				basic_point.z = radius * cosf(pcl::deg2rad(angle1));
				cloud->points.push_back(basic_point);
			}
		}
	}

	// 构造立方体
	float cube_len = 2;
	for (float x = 0; x < cube_len; x += 0.1)
	{
		for (float y = 0; y < cube_len; y += 0.1)
		{
			for (float z = 0; z < cube_len; z += 0.1)
			{
				pcl::PointXYZ basic_point;

				// 沿着向量(2.5, 2.5, 2.5)平移
				basic_point.x = x + 2.5;
				basic_point.y = y + 2.5;
				basic_point.z = z + 2.5;
				cloud->points.push_back(basic_point);
			}
		}
	}

	// 构造圆形平面
	float R = 1;
	for (float radius = 0; radius < R; radius += 0.05)
	{
		for (float r = 0; r < radius; r += 0.05)
		{
			for (float ang = 0; ang <= 360.0; ang += 5.0)
			{
				pcl::PointXYZ basic_point;

				basic_point.x = radius * sinf(pcl::deg2rad(ang)) + 3;
				basic_point.y = radius * cosf(pcl::deg2rad(ang)) + 3;
				basic_point.z = -3;
				cloud->points.push_back(basic_point);
			}
		}
	}

	cloud->width = (int)cloud->points.size();
	cloud->height = 1;

	// ------------------------------K均值聚类-----------------------------
	pcl::StopWatch time;
	int clusterNum = 3; // 聚类个数
	int maxIter = 50;   // 最大迭代次数
	KMeans kmeans(clusterNum, maxIter);
	std::vector<pcl::Indices> cluster_indices;
	kmeans.extract(cloud, cluster_indices);
	cout << "聚类的个数为：" << cluster_indices.size() << endl;
	cout << "代码运行时间:" << time.getTimeSeconds() << "秒" << endl;

	// ---------------------------聚类结果分类保存--------------------------
	int begin = 1;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr dbscan_all_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	for (auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
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
		pcl::io::savePCDFileBinary("kmeans" + std::to_string(begin) + ".pcd", *cloud_dbscan);
		begin++;

	}

	return 0;
}

