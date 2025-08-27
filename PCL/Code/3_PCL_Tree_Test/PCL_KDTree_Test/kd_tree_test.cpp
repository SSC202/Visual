#include <pcl/point_cloud.h>					// ��������
#include <pcl/kdtree/kdtree_flann.h>			// KDtree��ض���
#include <pcl/visualization/cloud_viewer.h>		// ���ӻ���ض���

#include <iostream>
#include <vector>
#include <ctime>

#include <boost/thread/thread.hpp>

using namespace std;

int main(int argc, char** argv)
{

	// ʹ��ϵͳʱ�������������
	srand(time(NULL));

	// ����һ��PointXYZ���͵���ָ��
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	// ��ʼ����������
	cloud->width = 1000;// ��Ϊ1000
	cloud->height = 1;//��Ϊ1��˵��Ϊ�������
	cloud->points.resize(cloud->width * cloud->height);

	// ʹ��������������
	for (size_t i = 0; i < cloud->size(); ++i)
	{
		cloud->points[i].x = 1024.0f * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].y = 1024.0f * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].z = 1024.0f * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].b = 0;
		cloud->points[i].g = 255;
		cloud->points[i].r = 0;
	}

	// ���� k-d tree
	pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;

	// ���õ�������,����cloud������
	kdtree.setInputCloud(cloud);

	// ���ñ�������,����������
	pcl::PointXYZRGB searchPoint;
	searchPoint.x = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.y = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.z = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.b = 0;
	searchPoint.g = 0;
	searchPoint.r = 255;

	// ��ʼ KNN ����,K����Ϊ10
	int K = 10;

	// ���ڰ뾶����������
	float radius = 256.0f * rand() / (RAND_MAX + 1.0f);

	// �洢�������
	vector<int> pointIdxRadiusSearch;
	vector<float> pointRadiusSquaredDistance;

	// �洢�������
	vector<int> pointIdxNKNSearch(K);			// �����±�
	vector<float> pointNKNSquaredDistance(K);	// ��������ƽ��

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
			// ��ѯ�������ڵĵ���ɫ
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

	// ����������
	if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
	{
		for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
		{
			cout << "    " << cloud->points[pointIdxRadiusSearch[i]].x
				<< " " << cloud->points[pointIdxRadiusSearch[i]].x
				<< " " << cloud->points[pointIdxRadiusSearch[i]].z
				<< "( squared distance: " << pointRadiusSquaredDistance[i] << " )" << endl;
			// ��ѯ�������ڵĵ���ɫ
			cloud->points[pointIdxRadiusSearch[i]].r = 255;
			cloud->points[pointIdxRadiusSearch[i]].g = 0;
			cloud->points[pointIdxRadiusSearch[i]].b = 255;
		}
	}

	// ���ӻ�
	pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	viewer.setBackgroundColor(0.0, 0.0, 0.0);
	viewer.addPointCloud<pcl::PointXYZRGB>(cloud, "cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");

	while (!viewer.wasStopped()) {
		viewer.spinOnce();
		boost::this_thread::sleep(boost::posix_time::microseconds(10000));
	}

	return 0;
}
