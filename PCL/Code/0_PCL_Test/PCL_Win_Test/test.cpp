#include <vtkNew.h>
#include <vtkPointSource.h>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_types.h>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>

int main(int, char* [])
{
	// -------------------------����λ�������ϵĵ���---------------------------
	vtkNew<vtkPointSource> pointSource;
	pointSource->SetCenter(0.0, 0.0, 0.0);
	pointSource->SetNumberOfPoints(5000);
	pointSource->SetRadius(5.0);
	pointSource->SetDistributionToShell();  // ���õ�ֲ��������ϡ�
	pointSource->Update();
	// ---------------------------תΪPCD���Ʋ�����----------------------------
	vtkSmartPointer<vtkPolyData> polydata = pointSource->GetOutput(); // ��ȡVTK�е�PolyData����
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::vtkPolyDataToPointCloud(polydata, *cloud);
	pcl::PCDWriter w;
	w.writeBinaryCompressed("sphere.pcd", *cloud);
	// -------------------------------������ӻ�-------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->setWindowName(u8"�������ε���");
	pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildColor(cloud, "z"); // ����z�ֶν�����Ⱦ
	viewer->addPointCloud<pcl::PointXYZ>(cloud, fildColor, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud"); // ���õ��ƴ�С

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

	return 0;
}

