#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h> //����������
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

using namespace std;

int main(int argc, char** argv) {
	// ���ص�������
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile<pcl::PointXYZ>("test.pcd", *cloud);							// ��ȡ����
	cout << "Loaded " << cloud->points.size() << " points." << endl;					// ��ʾ��ȡ���Ƶĸ���
	// ������Ƶķ��� 
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	n.setInputCloud(cloud);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	n.setSearchMethod(tree);															// ���������������ʽ
	// n.setRadiusSearch (0.03);														// ����KD�������뾶
	n.setKSearch(10);
	// ����һ���µĵ��ƴ��溬�з��ߵ�ֵ
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	n.compute(*normals);																// ����������ߵ�ֵ

	// �����ʼ���
	pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> p;
	p.setInputCloud(cloud);																// �ṩԭʼ����(û�з���)
	p.setInputNormals(normals);															// Ϊ�����ṩ����
	p.setSearchMethod(tree);															// ʹ���뷨�߹�����ͬ��KdTree
	// p.setRadiusSearch(1.0);
	p.setKSearch(10);
	// ����������
	pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr pri(new pcl::PointCloud<pcl::PrincipalCurvatures>());
	p.compute(*pri);
	cout << "output points.size: " << pri->points.size() << endl;
	// ��ʾ�ͼ�����0��������ʡ�
	cout << "���������;" << pri->points[0].pc1 << endl;// ����������
	cout << "��С������:" << pri->points[0].pc2 << endl;// �����С����
	//��������ʷ����������ֵ��Ӧ������������
	cout << "�����ʷ���;" << endl;
	cout << pri->points[0].principal_curvature_x << endl;
	cout << pri->points[0].principal_curvature_y << endl;
	cout << pri->points[0].principal_curvature_z << endl;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Normal viewer"));
	viewer->setBackgroundColor(0.3, 0.3, 0.3);     //���ñ�����ɫ
	viewer->addText("Curvatures", 10, 10, "text"); //������ʾ����
	viewer->setWindowName("Curvatures");           //���ô�������

	viewer->addCoordinateSystem(0.1);              //�������ϵ

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 225, 0); //���õ�����ɫ

	viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, "cloud"); //��ӵ��Ƶ����ӻ�����
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud"); //���õ��ƴ�С

	//�����Ҫ��ʾ�ĵ��Ʒ���cloudΪԭʼ����ģ�ͣ�normalΪ������Ϣ��20��ʾ��Ҫ��ʾ����ĵ��Ƽ������ÿ20������ʾһ�η���2��ʾ���򳤶ȡ�
	viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 20, 2, "normals");
	// �����Ҫ��ʾ�ĵ��������ʡ�cloudΪԭʼ����ģ�ͣ�normalΪ������Ϣ��priΪ���������ʣ�
	// 10��ʾ��Ҫ��ʾ���ʵĵ��Ƽ������ÿ10������ʾһ�������ʣ�10��ʾ���򳤶ȡ�
	// ĿǰaddPointCloudPrincipalCurvaturesֻ����<pcl::PointXYZ>��<pcl::Normal>����������δ��ʵ�����ʵĿ��ӻ���
	viewer->addPointCloudPrincipalCurvatures<pcl::PointXYZ, pcl::Normal>(cloud, normals, pri, 10, 10, "Curvatures");

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

	return 0;
}

