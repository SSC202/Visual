#include <iostream>
#include <thread>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>

using namespace std;

/**
 * @brief Visualize SIFT keypoint extraction results with normal-based detection
 * 使用基于法向梯度的SIFT关键点提取结果可视化
 */
void visualize_sift_normals_keypoints(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_xyz,
                                      pcl::PointCloud<pcl::PointXYZ>::Ptr &keypoints,
                                      double normal_time,
                                      double sift_time,
                                      const std::string &window_title = "SIFT Keypoint Detection with Normals")
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0.05, 0.05, 0.15); // 深蓝色背景

    // 创建两个视口
    int v1(0), v2(1);
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1); // 左：原始点云
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2); // 右：关键点提取结果

    // 设置各视口背景色
    viewer->setBackgroundColor(0.1, 0.1, 0.2, v1);  // 深蓝
    viewer->setBackgroundColor(0.1, 0.15, 0.1, v2); // 深绿

    // 颜色处理器
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        cloud_red(cloud_xyz, 255, 0, 0); // 原始点云 - 红色

    // 视口1: 原始点云
    viewer->addPointCloud<pcl::PointXYZ>(cloud_xyz, cloud_red, "original_cloud", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original_cloud", v1);

    viewer->addText("Original Point Cloud", 10, 20, 16, 1, 1, 1, "original_text", v1);
    viewer->addText("Red: All Points", 10, 40, 14, 1, 0, 0, "cloud_red_text", v1);

    std::string cloud_info = "Total Points: " + std::to_string(cloud_xyz->size());
    std::string file_info = "File: bunny.pcd";
    viewer->addText(cloud_info, 10, 60, 12, 1, 1, 1, "cloud_info", v1);
    viewer->addText(file_info, 10, 75, 12, 1, 1, 1, "file_info", v1);

    viewer->addCoordinateSystem(0.1, "axis_v1", v1);

    // 视口2: 关键点提取结果
    // 首先显示原始点云作为背景参考
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        cloud_gray(cloud_xyz, 100, 100, 100); // 原始点云 - 灰色作为背景
    viewer->addPointCloud<pcl::PointXYZ>(cloud_xyz, cloud_gray, "background_cloud", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "background_cloud", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, "background_cloud", v2);

    // 检查是否有关键点并添加
    if (keypoints->size() > 0)
    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            keypoints_green(keypoints, 0, 255, 0); // 关键点 - 绿色
        viewer->addPointCloud<pcl::PointXYZ>(keypoints, keypoints_green, "keypoints_cloud", v2);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "keypoints_cloud", v2);
    }

    viewer->addText("SIFT Keypoint Detection", 10, 20, 16, 1, 1, 1, "keypoints_text", v2);
    viewer->addText("Based on Normal Gradient", 10, 35, 12, 1, 1, 1, "method_text", v2);

    if (keypoints->size() == 0)
    {
        viewer->addText("NO KEYPOINTS DETECTED!", 10, 50, 14, 1, 0, 0, "no_keypoints_text", v2);
    }
    else
    {
        viewer->addText("Green: Extracted Keypoints", 10, 50, 14, 0, 1, 0, "keypoints_green_text", v2);
    }

    std::string keypoints_info = "Keypoints: " + std::to_string(keypoints->size());
    std::string keypoints_ratio = "Keypoint Ratio: " +
                                  std::to_string(static_cast<double>(keypoints->size()) / cloud_xyz->size() * 100).substr(0, 5) + "%";
    std::string normal_time_info = "Normal Computation: " + std::to_string(normal_time) + " ms";
    std::string sift_time_info = "SIFT Computation: " + std::to_string(sift_time) + " ms";

    viewer->addText(keypoints_info, 10, 70, 12, 1, 1, 1, "keypoints_info", v2);
    viewer->addText(keypoints_ratio, 10, 85, 12, 1, 1, 1, "keypoints_ratio", v2);
    viewer->addText(normal_time_info, 10, 100, 12, 1, 1, 1, "normal_time_info", v2);
    viewer->addText(sift_time_info, 10, 115, 12, 1, 1, 1, "sift_time_info", v2);

    viewer->addCoordinateSystem(0.1, "axis_v2", v2);

    // 设置相机参数
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 0.3, 0, 0, 0, 0, 1, 0);

    // 主循环
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main()
{
    cout << "SIFT Keypoint Extraction Analysis" << endl;
    cout << "==================================================" << endl;

    auto total_start_time = std::chrono::steady_clock::now();

    // 加载点云数据
    auto load_start_time = std::chrono::steady_clock::now();
    string filename = "cloud.pcd";

    cout << "Reading " << filename << endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud_xyz);

    auto load_end_time = std::chrono::steady_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end_time - load_start_time);

    cout << "Point Cloud Information:" << endl;
    cout << "  Loaded " << cloud_xyz->size() << " points" << endl;
    cout << "  Loading time: " << load_duration.count() << " ms" << endl;

    // SIFT算法参数
    cout << "\nSIFT Algorithm Parameters:" << endl;
    const float min_scale = 0.05f;     // 设置尺度空间中最小尺度的标准偏差
    const int n_octaves = 3;           // 设置尺度空间层数，越小则特征点越多
    const int n_scales_per_octave = 4; // 设置尺度空间中计算的尺度个数
    const float min_contrast = 0.001f; // 设置限制关键点检测的阈值

    cout << "  Min scale: " << min_scale << endl;
    cout << "  Number of octaves: " << n_octaves << endl;
    cout << "  Scales per octave: " << n_scales_per_octave << endl;
    cout << "  Minimum contrast: " << min_contrast << endl;

    // 计算cloud_xyz的法向量和表面曲率
    cout << "\nComputing Surface Normals:" << endl;
    auto normal_start_time = std::chrono::steady_clock::now();

    pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_n(new pcl::search::KdTree<pcl::PointXYZ>());

    ne.setInputCloud(cloud_xyz);
    ne.setSearchMethod(tree_n);
    ne.setRadiusSearch(0.1);
    ne.compute(*cloud_normals);

    auto normal_end_time = std::chrono::steady_clock::now();
    auto normal_duration = std::chrono::duration_cast<std::chrono::milliseconds>(normal_end_time - normal_start_time);

    cout << "  Normals computed: " << cloud_normals->size() << endl;
    cout << "  Processing time: " << normal_duration.count() << " ms" << endl;
    cout << "  Search radius: 0.1" << endl;

    // 从cloud_xyz复制xyz信息，并将其添加到cloud_normals中
    for (std::size_t i = 0; i < cloud_normals->size(); ++i)
    {
        (*cloud_normals)[i].x = (*cloud_xyz)[i].x;
        (*cloud_normals)[i].y = (*cloud_xyz)[i].y;
        (*cloud_normals)[i].z = (*cloud_xyz)[i].z;
    }

    // 使用法线值作为强度变量估计SIFT关键点
    cout << "\nComputing SIFT Keypoints:" << endl;
    auto sift_start_time = std::chrono::steady_clock::now();

    pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
    pcl::PointCloud<pcl::PointWithScale> result;
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>());

    sift.setSearchMethod(tree);
    sift.setScales(min_scale, n_octaves, n_scales_per_octave);
    sift.setMinimumContrast(min_contrast);
    sift.setInputCloud(cloud_normals);
    sift.compute(result);

    auto sift_end_time = std::chrono::steady_clock::now();
    auto sift_duration = std::chrono::duration_cast<std::chrono::milliseconds>(sift_end_time - sift_start_time);

    cout << "  SIFT keypoints extracted: " << result.size() << endl;
    cout << "  Processing time: " << sift_duration.count() << " ms" << endl;

    // 将点云转为PointXYZ进行可视化
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(result, *keypoints_xyz);

    cout << "\nKeypoint Statistics:" << endl;
    cout << "  Keypoints in result: " << result.size() << endl;
    cout << "  Keypoints in cloud_temp: " << keypoints_xyz->size() << endl;

    if (keypoints_xyz->size() > 0)
    {
        cout << "  Keypoint ratio: " << static_cast<double>(keypoints_xyz->size()) / cloud_xyz->size() * 100 << "%" << endl;
    }

    // 分析总结
    auto total_end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);

    cout << "\nAnalysis Summary:" << endl;
    cout << "  Total processing time: " << total_duration.count() << " ms" << endl;
    cout << "  Breakdown:" << endl;
    cout << "    - Point cloud loading: " << load_duration.count() << " ms" << endl;
    cout << "    - Normal computation: " << normal_duration.count() << " ms" << endl;
    cout << "    - SIFT keypoint extraction: " << sift_duration.count() << " ms" << endl;
    cout << "  Original points: " << cloud_xyz->size() << endl;
    cout << "  Extracted keypoints: " << keypoints_xyz->size() << endl;
    cout << "  Method: Based on normal gradient (法向梯度)" << endl;

    // 调用可视化函数
    visualize_sift_normals_keypoints(cloud_xyz, keypoints_xyz,
                                     normal_duration.count(),
                                     sift_duration.count(),
                                     "SIFT Keypoint Detection with Normals - Bunny Dataset");

    return 0;
}