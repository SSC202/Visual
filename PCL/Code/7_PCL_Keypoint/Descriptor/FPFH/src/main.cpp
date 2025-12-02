#include <iostream>
#include <thread>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/search/kdtree.h>

using namespace std;

int main()
{
    cout << "FPFH Feature Extraction Analysis" << endl;
    cout << "==================================" << endl;

    auto total_start_time = std::chrono::steady_clock::now();

    // 读取点云
    auto load_start_time = std::chrono::steady_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PCDReader reader;
    reader.read("cloud.pcd", *cloud);

    auto load_end_time = std::chrono::steady_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end_time - load_start_time);

    cout << "Point Cloud Information:" << endl;
    cout << "  Loaded " << cloud->size() << " points" << endl;
    cout << "  Loading time: " << load_duration.count() << " ms" << endl;

    // 计算法向量
    cout << "\nComputing Surface Normals (OMP accelerated):" << endl;
    auto normal_start_time = std::chrono::steady_clock::now();

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setInputCloud(cloud);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setNumberOfThreads(4);
    normal_estimator.setKSearch(10);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    normal_estimator.compute(*normals);

    auto normal_end_time = std::chrono::steady_clock::now();
    auto normal_duration = std::chrono::duration_cast<std::chrono::milliseconds>(normal_end_time - normal_start_time);

    cout << "  Normals computed: " << normals->size() << endl;
    cout << "  Processing time: " << normal_duration.count() << " ms" << endl;
    cout << "  KSearch value: 10" << endl;
    cout << "  Threads: 4" << endl;

    // 计算 FPFH
    cout << "\nComputing FPFH Features:" << endl;
    auto fpfh_start_time = std::chrono::steady_clock::now();

    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>());
    fpfh.setSearchMethod(tree2);
    fpfh.setRadiusSearch(0.1);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_features(new pcl::PointCloud<pcl::FPFHSignature33>());
    fpfh.compute(*fpfh_features);

    auto fpfh_end_time = std::chrono::steady_clock::now();
    auto fpfh_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fpfh_end_time - fpfh_start_time);

    cout << "  FPFH features computed: " << fpfh_features->size() << endl;
    cout << "  Processing time: " << fpfh_duration.count() << " ms" << endl;
    cout << "  Search radius: 0.1" << endl;
    cout << "  Feature dimension: 33" << endl;

    // 直方图可视化
    cout << "\nOpening FPFH feature histogram..." << endl;

    // 显示所有点的平均直方图
    pcl::visualization::PCLPlotter plotter;
    plotter.addFeatureHistogram(*fpfh_features, 300);
    plotter.plot();

    // 分析总结
    auto total_end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);

    cout << "\nAnalysis Summary:" << endl;
    cout << "  Total processing time: " << total_duration.count() << " ms" << endl;
    cout << "  Point cloud size: " << cloud->size() << endl;
    cout << "  Normals computed: " << normals->size() << endl;
    cout << "  FPFH features extracted: " << fpfh_features->size() << endl;
    cout << "  Normal estimation method: OMP accelerated" << endl;
    cout << "  Feature type: FPFHSignature33" << endl;

    return 0;
}