#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h> //计算主曲率
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

using namespace std;

int main(int argc, char** argv) {
	// 加载点云数据
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile<pcl::PointXYZ>("test.pcd", *cloud);							// 读取点云
	cout << "Loaded " << cloud->points.size() << " points." << endl;					// 显示读取点云的个数
	// 计算点云的法线 
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	n.setInputCloud(cloud);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	n.setSearchMethod(tree);															// 设置邻域点搜索方式
	// n.setRadiusSearch (0.03);														// 设置KD树搜索半径
	n.setKSearch(10);
	// 定义一个新的点云储存含有法线的值
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	n.compute(*normals);																// 计算出来法线的值

	// 主曲率计算
	pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> p;
	p.setInputCloud(cloud);																// 提供原始点云(没有法线)
	p.setInputNormals(normals);															// 为点云提供法线
	p.setSearchMethod(tree);															// 使用与法线估算相同的KdTree
	// p.setRadiusSearch(1.0);
	p.setKSearch(10);
	// 计算主曲率
	pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr pri(new pcl::PointCloud<pcl::PrincipalCurvatures>());
	p.compute(*pri);
	cout << "output points.size: " << pri->points.size() << endl;
	// 显示和检索第0点的主曲率。
	cout << "最大主曲率;" << pri->points[0].pc1 << endl;// 输出最大曲率
	cout << "最小主曲率:" << pri->points[0].pc2 << endl;// 输出最小曲率
	//输出主曲率方向（最大特征值对应的特征向量）
	cout << "主曲率方向;" << endl;
	cout << pri->points[0].principal_curvature_x << endl;
	cout << pri->points[0].principal_curvature_y << endl;
	cout << pri->points[0].principal_curvature_z << endl;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Normal viewer"));
	viewer->setBackgroundColor(0.3, 0.3, 0.3);     //设置背景颜色
	viewer->addText("Curvatures", 10, 10, "text"); //设置显示文字
	viewer->setWindowName("Curvatures");           //设置窗口名字

	viewer->addCoordinateSystem(0.1);              //添加坐标系

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 225, 0); //设置点云颜色

	viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, "cloud"); //添加点云到可视化窗口
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud"); //设置点云大小

	//添加需要显示的点云法向。cloud为原始点云模型，normal为法向信息，20表示需要显示法向的点云间隔，即每20个点显示一次法向，2表示法向长度。
	viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 20, 2, "normals");
	// 添加需要显示的点云主曲率。cloud为原始点云模型，normal为法向信息，pri为点云主曲率，
	// 10表示需要显示曲率的点云间隔，即每10个点显示一次主曲率，10表示法向长度。
	// 目前addPointCloudPrincipalCurvatures只接受<pcl::PointXYZ>和<pcl::Normal>两个参数，未能实现曲率的可视化。
	viewer->addPointCloudPrincipalCurvatures<pcl::PointXYZ, pcl::Normal>(cloud, normals, pri, 10, 10, "Curvatures");

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

	return 0;
}

