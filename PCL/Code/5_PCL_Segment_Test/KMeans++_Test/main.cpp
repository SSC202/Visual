#include "Kmeans.h"
#include <pcl/io/pcd_io.h>
#include <pcl/common/time.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;

int main()
{
	// -------------------------------���ص���-----------------------------
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("test.pcd", *cloud) == -1)
	{
		PCL_ERROR("��ȡԴ�����ʧ�� \n");
		return (-1);
	}
	cout << "�ӵ����ж�ȡ " << cloud->size() << " ����" << endl;
	// ------------------------------K��ֵ����-----------------------------
	pcl::StopWatch time;
	int clusterNum = 3; // �������
	int maxIter = 50;   // ����������
	KMeans kmeans(clusterNum, maxIter);
	std::vector<pcl::Indices> cluster_indices;
	kmeans.extract(cloud, cluster_indices);
	cout << "����ĸ���Ϊ��" << cluster_indices.size() << endl;
	cout << "��������ʱ��:" << time.getTimeSeconds() << "��" << endl;

	// ---------------------------���������ౣ��--------------------------
	int begin = 1;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr all_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	for (auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		// ��ȡÿһ����������ŵĵ�
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_kmeans(new pcl::PointCloud<pcl::PointXYZRGB>);
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
			cloud_kmeans->points.push_back(point_db);
		}
		// ���������ౣ��
		pcl::io::savePCDFileBinary("kmeans" + std::to_string(begin) + ".pcd", *cloud_kmeans);
		begin++;

		*all_cloud += *cloud_kmeans;
	}

	pcl::visualization::CloudViewer viewer("Kmeans cloud viewer.");
	viewer.showCloud(all_cloud);
	while (!viewer.wasStopped())
	{

	}
	return 0;
}

