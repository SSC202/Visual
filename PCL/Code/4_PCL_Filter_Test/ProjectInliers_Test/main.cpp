#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;

int main()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);					//ԭʼ����
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected(new pcl::PointCloud<pcl::PointXYZ>);		//ͶӰ����

	// �����������
	cout << "->���ڶ������..." << endl;
	pcl::PCDReader reader;
	reader.read("test.pcd", *cloud);
	cout << "\t\t<���������Ϣ>\n" << *cloud << endl;

	// ������ģ��ͶӰ
	cout << "->����ƽ��ģ��ͶӰ..." << endl;
	// ����ƽ��
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	coefficients->values.resize(4);	//����ģ��ϵ���Ĵ�С
	coefficients->values[0] = 1.0;	//xϵ��
	coefficients->values[1] = 1.0;	//yϵ��
	coefficients->values[2] = 1.0;	//zϵ��
	coefficients->values[3] = 0.0;	//������

	// ͶӰ�˲�
	pcl::ProjectInliers<pcl::PointXYZ> proj;//����ͶӰ�˲�������
	proj.setModelType(pcl::SACMODEL_PLANE);	//���ö����Ӧ��ͶӰģ��
	proj.setInputCloud(cloud);				//�����������
	proj.setModelCoefficients(coefficients);//����ģ�Ͷ�Ӧ��ϵ��
	proj.filter(*cloud_projected);			//ִ��ͶӰ�˲����洢�����cloud_projected

	// �����˲�����
	cout << "->���ڱ���ͶӰ����..." << endl;
	pcl::PCDWriter writer;
	writer.write("proj_PLANE.pcd", *cloud_projected, true);
	cout << "\t\t<���������Ϣ>\n" << *cloud_projected << endl;

	// ���ӻ�
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("�˲�ǰ��Ա�"));

	// ��ͼ1
	int v1(0);
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1); //���õ�һ���ӿ���X�ᡢY�����Сֵ�����ֵ��ȡֵ��0-1֮��
	viewer->setBackgroundColor(0, 0, 0, v1); //���ñ�����ɫ��0-1��Ĭ�Ϻ�ɫ��0��0��0��
	viewer->addText("befor_filtered", 10, 10, "v1_text", v1);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, "befor_filtered_cloud", v1);

	// ��ͼ2
	int v2(0);
	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	viewer->setBackgroundColor(0.3, 0.3, 0.3, v2);
	viewer->addText("after_filtered", 10, 10, "v2_text", v2);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_projected, "after_filtered_cloud", v2);

	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "befor_filtered_cloud", v1);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "befor_filtered_cloud", v1);

	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "after_filtered_cloud", v2);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "after_filtered_cloud", v2);

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
	}
	return 0;
}

