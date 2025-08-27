#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>				//ʹ��OMP��Ҫ��ӵ�ͷ�ļ�
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

using namespace std;
int main()
{
	// ���ص�������
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("test.pcd", *cloud) == -1)
	{
		PCL_ERROR("Could not read file\n");
	}

	// ���㷨�� 
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;									// OMP����
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>()); // ����kdtree�����н��ڵ㼯����
	n.setNumberOfThreads(10);																// ����openMP���߳���
	// n.setViewPoint(0,0,0);																// �����ӵ㣬Ĭ��Ϊ��0��0��0��
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(10);																		// ���Ʒ������ʱ����Ҫ���ѵĽ��ڵ��С
	// n.setRadiusSearch(0.03);																// �뾶����
	n.compute(*normals);																	// ��ʼ���з���������

	// ���ӻ�
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Normal viewer"));
	// viewer->initCameraParameters();														// ���������������ʹ�û���Ĭ�ϵĽǶȺͷ���۲����
	viewer->setBackgroundColor(0.3, 0.3, 0.3);
	viewer->addText("faxian", 10, 10, "text");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 225, 0);
	viewer->addCoordinateSystem(0.1);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, "sample cloud");


	// �����Ҫ��ʾ�ĵ��Ʒ���cloudΪԭʼ����ģ�ͣ�normalΪ������Ϣ��20��ʾ��Ҫ��ʾ����ĵ��Ƽ������ÿ20������ʾһ�η���0.02��ʾ���򳤶ȡ�
	viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 20, 0.02, "normals");
	// ���õ��ƴ�С
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

	return 0;
}
