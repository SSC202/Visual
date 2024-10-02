#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/cloud_viewer.h>
 
#include <iostream>
#include <vector>
#include <ctime>

#include <X11/Xlib.h>
 
using namespace std;
 
int main(int argc, char** argv)
{
	XInitThreads();

	// 使用系统时间做随机数种子
	srand(time(NULL));
	// 创建一个PointXYZ类型点云指针
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
 
	// 初始化点云数据
	cloud->width = 1000;// 宽为1000
	cloud->height = 1;//高为1，说明为无序点云
	cloud->points.resize(cloud->width * cloud->height);

	// 使用随机数填充数据
	for (size_t i = 0; i < cloud->size(); ++i)
	{
		// PointCloud类中对[]操作符进行了重载，返回的是对points的引用
		// (*cloud)[i].x 等同于 cloud->points[i].x
		(*cloud)[i].x = 1024.0f * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].y = 1024.0f * rand() / (RAND_MAX + 1.0f);//推进写法
		cloud->points[i].z = 1024.0f * rand() / (RAND_MAX + 1.0f);//推进写法
	}
 
	// 创建kd树对象
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
 
	// 设置点云输入,将在cloud中搜索
	kdtree.setInputCloud(cloud);
 
	// 设置被搜索点,用随机数填充
	pcl::PointXYZ searchPoint;
	searchPoint.x = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.y = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.z = 1024.0f * rand() / (RAND_MAX + 1.0f);
 
	//基于半径的邻域搜索
	//搜索结果保存在两个数组中，一个是下标，一个是距离
	vector<int> pointIdxRadiusSearch;
	vector<float> pointRadiusSquaredDistance;
 
	//设置搜索半径，随机值
	float radius = 256.0f* rand() / (RAND_MAX + 1.0f);
 
	cout << "Neighbors within radius search at (" << searchPoint.x
		<< " " << searchPoint.y
		<< " " << searchPoint.z
		<< ") with radius=" << radius << endl;
 
	/**
	 * 如果在指定的半径内返回超过0个邻居，将打印出这些存储在向量中的点的坐标。
	 */
	if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
	{
		for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
		{
			cout << "    " << cloud->points[pointIdxRadiusSearch[i]].x
				<< " " << cloud->points[pointIdxRadiusSearch[i]].x
				<< " " << cloud->points[pointIdxRadiusSearch[i]].z
				<< "( squared distance: " << pointRadiusSquaredDistance[i] << " )" << endl;
		}
	}

    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    viewer.setBackgroundColor(0.0, 0.0, 0.5);
    viewer.addPointCloud<pcl::PointXYZ>(cloud, "cloud");

    pcl::PointXYZ originPoint(0.0, 0.0, 0.0);
    // 添加从原点到搜索点的线段
    viewer.addLine(originPoint, searchPoint);
    // 添加一个以搜索点为圆心，搜索半径为半径的球体
    viewer.addSphere(searchPoint, radius, "sphere", 0);
    // 添加一个放到200倍后的坐标系
    viewer.addCoordinateSystem(200);

    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }

	return 0;
}
