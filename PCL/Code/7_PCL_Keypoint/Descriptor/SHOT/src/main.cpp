#include <iostream>
#include <thread>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/search/kdtree.h>

using namespace std;

int main()
{
    cout << "SHOT Descriptor Extraction Analysis" << endl;
    cout << "====================================" << endl;

    auto total_start_time = std::chrono::steady_clock::now();

    // 加载点云数据
    auto load_start_time = std::chrono::steady_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    pcl::io::loadPCDFile<pcl::PointXYZ>("cloud.pcd", *cloud);
    
    auto load_end_time = std::chrono::steady_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end_time - load_start_time);

    cout << "Point Cloud Information:" << endl;
    cout << "  Loaded " << cloud->size() << " points" << endl;
    cout << "  Loading time: " << load_duration.count() << " ms" << endl;

    // 计算法线
    cout << "\nComputing Surface Normals (OMP accelerated):" << endl;
    auto normal_start_time = std::chrono::steady_clock::now();
    
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    
    normal_estimator.setNumberOfThreads(4);
    normal_estimator.setInputCloud(cloud);
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setKSearch(30);
    normal_estimator.compute(*normals);
    
    auto normal_end_time = std::chrono::steady_clock::now();
    auto normal_duration = std::chrono::duration_cast<std::chrono::milliseconds>(normal_end_time - normal_start_time);
    
    cout << "  Normals computed: " << normals->size() << endl;
    cout << "  Processing time: " << normal_duration.count() << " ms" << endl;
    cout << "  KSearch value: 30" << endl;
    cout << "  Threads: 4" << endl;

    // 均匀采样提取关键点
    cout << "\nUniform Sampling Keypoints:" << endl;
    auto sampling_start_time = std::chrono::steady_clock::now();
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::UniformSampling<pcl::PointXYZ> uniform_sampling;
    
    uniform_sampling.setInputCloud(cloud);
    uniform_sampling.setRadiusSearch(0.5f);
    uniform_sampling.filter(*cloud_filtered);
    
    auto sampling_end_time = std::chrono::steady_clock::now();
    auto sampling_duration = std::chrono::duration_cast<std::chrono::milliseconds>(sampling_end_time - sampling_start_time);
    
    cout << "  Keypoints after uniform sampling: " << cloud_filtered->size() << endl;
    cout << "  Processing time: " << sampling_duration.count() << " ms" << endl;
    cout << "  Search radius: 0.5" << endl;
    cout << "  Keypoint ratio: " << static_cast<double>(cloud_filtered->size()) / cloud->size() * 100 << "%" << endl;

    // 为关键点计算描述子
    cout << "\nComputing SHOT Descriptors (OMP accelerated):" << endl;
    auto shot_start_time = std::chrono::steady_clock::now();
    
    pcl::PointCloud<pcl::SHOT352>::Ptr shot_descriptors(new pcl::PointCloud<pcl::SHOT352>());
    pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot352;
    
    shot352.setRadiusSearch(0.5);
    shot352.setInputCloud(cloud_filtered);
    shot352.setInputNormals(normals);
    shot352.setSearchSurface(cloud);
    shot352.compute(*shot_descriptors);
    
    auto shot_end_time = std::chrono::steady_clock::now();
    auto shot_duration = std::chrono::duration_cast<std::chrono::milliseconds>(shot_end_time - shot_start_time);
    
    cout << "  SHOT descriptors computed: " << shot_descriptors->size() << endl;
    cout << "  Processing time: " << shot_duration.count() << " ms" << endl;
    cout << "  Search radius: 0.5" << endl;
    cout << "  Feature dimension: 352" << endl;
    cout << "  Descriptor type: SHOT352" << endl;

    // 分析总结
    auto total_end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);

    cout << "\nAnalysis Summary:" << endl;
    cout << "  Total processing time: " << total_duration.count() << " ms" << endl;
    cout << "  Breakdown:" << endl;
    cout << "    - Point cloud loading: " << load_duration.count() << " ms" << endl;
    cout << "    - Normal computation: " << normal_duration.count() << " ms" << endl;
    cout << "    - Uniform sampling: " << sampling_duration.count() << " ms" << endl;
    cout << "    - SHOT descriptor extraction: " << shot_duration.count() << " ms" << endl;
    cout << "  Original points: " << cloud->size() << endl;
    cout << "  Keypoints (after uniform sampling): " << cloud_filtered->size() << endl;
    cout << "  SHOT descriptors extracted: " << shot_descriptors->size() << endl;
    cout << "  Normal estimation method: OMP accelerated" << endl;
    cout << "  Descriptor type: SHOT352" << endl;

    return 0;
}