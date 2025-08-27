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
	// -------------------------生成位于球面上的点云---------------------------
	vtkNew<vtkPointSource> pointSource;
	pointSource->SetCenter(0.0, 0.0, 0.0);
	pointSource->SetNumberOfPoints(5000);
	pointSource->SetRadius(5.0);
	pointSource->SetDistributionToShell();  // 设置点分布在球面上。
	pointSource->Update();
	// ---------------------------转为PCD点云并保存----------------------------
	vtkSmartPointer<vtkPolyData> polydata = pointSource->GetOutput(); // 获取VTK中的PolyData数据
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::vtkPolyDataToPointCloud(polydata, *cloud);
	pcl::PCDWriter w;
	w.writeBinaryCompressed("sphere.pcd", *cloud);
	// -------------------------------结果可视化-------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->setWindowName(u8"生成球形点云");
	pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildColor(cloud, "z"); // 按照z字段进行渲染
	viewer->addPointCloud<pcl::PointXYZ>(cloud, fildColor, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud"); // 设置点云大小

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

	return 0;
}

