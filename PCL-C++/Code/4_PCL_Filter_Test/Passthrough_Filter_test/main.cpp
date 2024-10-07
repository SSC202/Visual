#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;

int main()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);			//���˲�����
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);	//�˲������

	// �������
	cout << "->���ڶ������..." << endl;
	if (pcl::io::loadPCDFile("test.pcd", *cloud) < 0)
	{
		PCL_ERROR("�����ļ������ڣ�\n");
		system("pause");
		return -1;
	}
	cout << "\t\t<���������Ϣ>\n" << *cloud << endl;

	// ֱͨ�˲�
	cout << "->���ڽ���ֱͨ�˲�..." << endl;
	pcl::PassThrough<pcl::PointXYZ> pt;			// �����˲�������
	pt.setInputCloud(cloud);					// �����������
	pt.setFilterFieldName("x");					// �����˲������ֶ�
	pt.setFilterLimits(-0.1, 1);				// �����ֶι��˷�Χ
	// pt.setFilterLimitsNegative(true);		

	pt.filter(*cloud_filtered);					// ִ���˲��������˲������

	// �����˲������
	cout << "->���ڱ������...\n";
	if (cloud_filtered->empty())
	{
		PCL_ERROR("�������Ϊ��!\n");
		return -1;
	}
	else
	{
		pcl::io::savePCDFileASCII("filter.pcd", *cloud_filtered);
		cout << "\t\t<���������Ϣ>\n" << *cloud_filtered << endl;
	}

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
	viewer->addPointCloud<pcl::PointXYZ>(cloud_filtered, "after_filtered_cloud", v2);

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

