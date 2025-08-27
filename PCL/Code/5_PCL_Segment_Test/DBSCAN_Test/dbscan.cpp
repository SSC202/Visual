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
	// --------------------------------��ȡ����-----------------------------------
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	if (pcl::io::loadPCDFile<pcl::PointXYZ>("test.pcd", *cloud) < 0)
	{
		PCL_ERROR("���ƶ�ȡʧ�ܣ����� \n");
		return -1;
	}
	cout << "�ӵ��������ж�ȡ��" << cloud->points.size() << "����" << endl;
	// -------------------------------�ܶȾ���------------------------------------
	pcl::StopWatch time;
	vector<pcl::Indices> cluster_indices;
	dbscan(*cloud, cluster_indices, 1, 50); // 2��ʾ������������Ϊ2�ף�50��ʾ�������С������

	cout << "�ܶȾ���ĸ���Ϊ��" << cluster_indices.size() << endl;
	cout << "��������ʱ��:" << time.getTimeSeconds() << "��" << endl;
	// ---------------------------���������ౣ��--------------------------------
	int begin = 1;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr dbscan_all_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	for (vector<pcl::Indices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		// ��ȡÿһ����������ŵĵ�
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_dbscan(new pcl::PointCloud<pcl::PointXYZRGB>);
		// ͬһ�����Ÿ���ͬһ����ɫ
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
		// ���������ౣ��
		stringstream ss;
		ss << "dbscan_cluster_" << begin << ".pcd";
		pcl::PCDWriter writer;
		writer.write<pcl::PointXYZRGB>(ss.str(), *cloud_dbscan, true);
		begin++;

		*dbscan_all_cloud += *cloud_dbscan;
	}
	// -------------------------------���������ӻ�----------------------------------
	pcl::visualization::CloudViewer viewer("DBSCAN cloud viewer.");
	viewer.showCloud(dbscan_all_cloud);
	while (!viewer.wasStopped())
	{

	}
	return 0;
}

