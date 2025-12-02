#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/pfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/search/kdtree.h>

using namespace std;

int main()
{
    cout << "PFH Feature Extraction Analysis" << endl;
    cout << "==================================" << endl;

    auto total_start_time = std::chrono::steady_clock::now();

    // 读取点云
    auto load_start_time = std::chrono::steady_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile<pcl::PointXYZ>("cloud.pcd", *cloud);

    auto load_end_time = std::chrono::steady_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end_time - load_start_time);

    cout << "Point Cloud Information:" << endl;
    cout << "  Loaded " << cloud->points.size() << " points" << endl;
    cout << "  Loading time: " << load_duration.count() << " ms" << endl;

    // 计算法向量
    auto normal_start_time = std::chrono::steady_clock::now();

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setInputCloud(cloud);
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setRadiusSearch(0.1);

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    normal_estimator.compute(*cloud_normals);

    auto normal_end_time = std::chrono::steady_clock::now();
    auto normal_duration = std::chrono::duration_cast<std::chrono::milliseconds>(normal_end_time - normal_start_time);

    cout << "\nNormal Computation:" << endl;
    cout << "  Normals computed: " << cloud_normals->points.size() << endl;
    cout << "  Processing time: " << normal_duration.count() << " ms" << endl;

    // 计算 PFH
    auto pfh_start_time = std::chrono::steady_clock::now();

    pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;
    pfh.setInputCloud(cloud);
    pfh.setInputNormals(cloud_normals);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>());
    pfh.setSearchMethod(tree2);
    pfh.setRadiusSearch(0.2);

    pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_features(new pcl::PointCloud<pcl::PFHSignature125>());
    pfh.compute(*pfh_features);

    auto pfh_end_time = std::chrono::steady_clock::now();
    auto pfh_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pfh_end_time - pfh_start_time);

    cout << "\nPFH Feature Extraction:" << endl;
    cout << "  PFH features computed: " << pfh_features->points.size() << endl;
    cout << "  Processing time: " << pfh_duration.count() << " ms" << endl;

    // 直方图可视化
    cout << "\nOpening PFH feature histogram..." << endl;
    pcl::visualization::PCLPlotter plotter;
    plotter.addFeatureHistogram(*pfh_features, 300);
    plotter.plot();

    // 分析总结
    auto total_end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);

    cout << "\nAnalysis Summary:" << endl;
    cout << "  Total processing time: " << total_duration.count() << " ms" << endl;
    cout << "  Point cloud size: " << cloud->points.size() << endl;
    cout << "  Normals computed: " << cloud_normals->points.size() << endl;
    cout << "  PFH features extracted: " << pfh_features->points.size() << endl;

    return 0;
}