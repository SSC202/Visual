#include <iostream>
#include <thread>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/common.h>

using namespace std;

/**
 * @brief Visualize Harris keypoint extraction results using dual viewports
 * 使用双视口可视化Harris关键点提取结果
 */
void visualize_harris_keypoints(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                pcl::PointCloud<pcl::PointXYZ>::Ptr &keypoints,
                                double processing_time,
                                const std::string &window_title = "Harris 3D Keypoint Extraction")
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
        cloud_green(cloud, 0, 225, 0); // 原始点云 - 绿色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        keypoints_red(keypoints, 255, 0, 0); // 关键点 - 红色

    // 视口1: 原始点云
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_green, "original_cloud", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original_cloud", v1);

    viewer->addText("Original Point Cloud", 10, 20, 16, 1, 1, 1, "original_text", v1);
    viewer->addText("Green: All Points", 10, 40, 14, 0, 1, 0, "cloud_green_text", v1);

    std::string cloud_info = "Total Points: " + std::to_string(cloud->size());
    std::string file_info = "File: rops_cloud.pcd";
    viewer->addText(cloud_info, 10, 60, 12, 1, 1, 1, "cloud_info", v1);
    viewer->addText(file_info, 10, 75, 12, 1, 1, 1, "file_info", v1);

    viewer->addCoordinateSystem(0.1, "axis_v1", v1);

    // 视口2: 关键点提取结果
    // 首先显示原始点云作为背景参考
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        cloud_gray(cloud, 100, 100, 100); // 原始点云 - 灰色作为背景
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_gray, "background_cloud", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "background_cloud", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, "background_cloud", v2);

    // 添加关键点（红色）
    viewer->addPointCloud<pcl::PointXYZ>(keypoints, keypoints_red, "keypoints_cloud", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "keypoints_cloud", v2);

    viewer->addText("Harris 3D Keypoint Extraction", 10, 20, 16, 1, 1, 1, "keypoints_text", v2);
    viewer->addText("Red: Extracted Keypoints", 10, 40, 14, 1, 0, 0, "keypoints_red_text", v2);

    std::string keypoints_info = "Keypoints: " + std::to_string(keypoints->size());
    std::string keypoints_ratio = "Keypoint Ratio: " +
                                  std::to_string(static_cast<double>(keypoints->size()) / cloud->size() * 100).substr(0, 5) + "%";
    std::string time_info = "Processing Time: " + std::to_string(processing_time) + " ms";
    std::string save_info = "Saved to: keypointsrgb.pcd";
    viewer->addText(keypoints_info, 10, 60, 12, 1, 1, 1, "keypoints_info", v2);
    viewer->addText(keypoints_ratio, 10, 75, 12, 1, 1, 1, "keypoints_ratio", v2);
    viewer->addText(time_info, 10, 90, 12, 1, 1, 1, "time_info", v2);
    viewer->addText(save_info, 10, 105, 12, 0, 1, 1, "save_info", v2);

    viewer->addCoordinateSystem(0.1, "axis_v2", v2);

    // 设置相机参数
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 3, 0, 0, 0, 0, 1, 0); // 适应点云

    // 主循环
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main()
{
    cout << "Harris 3D Keypoint Extraction Analysis" << endl;
    auto total_start_time = std::chrono::steady_clock::now();

    // 加载点云数据
    auto load_start_time = std::chrono::steady_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("cloud.pcd", *cloud) == -1)
    {
        cerr << "Error: Could not read PCD file: cloud.pcd" << endl;
        cerr << "Please ensure the file exists in the current directory." << endl;
        return -1;
    }
    auto load_end_time = std::chrono::steady_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end_time - load_start_time);

    cout << "Point Cloud Information:" << endl;
    cout << "  Loaded " << cloud->points.size() << " points" << endl;
    cout << "  Loading time: " << load_duration.count() << " ms" << endl;

    // Harris关键点提取
    cout << "Harris Keypoint Extraction:" << endl;
    auto harris_start_time = std::chrono::steady_clock::now();

    pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI> harris;
    pcl::PointCloud<pcl::PointXYZI>::Ptr harris_keypoints(new pcl::PointCloud<pcl::PointXYZI>);

    // Harris参数设置
    harris.setInputCloud(cloud);
    harris.setMethod(harris.LOWE);    // 设置要计算响应的方法
    harris.setRadius(10);             // 设置法线估计和非极大值抑制的半径
    harris.setRadiusSearch(10);       // 设置用于关键点检测的最近邻居的球半径
    harris.setNonMaxSupression(true); // 应用非最大值抑制
    harris.setThreshold(0.001);       // 设置角点检测阈值
    harris.setRefine(true);           // 检测到的关键点需要细化，设置为true时，关键点为点云中的点
    harris.setNumberOfThreads(6);     // 初始化调度程序并设置要使用的线程数

    harris.compute(*harris_keypoints);

    auto harris_end_time = std::chrono::steady_clock::now();
    auto harris_duration = std::chrono::duration_cast<std::chrono::milliseconds>(harris_end_time - harris_start_time);

    cout << "  Harris keypoints extracted: " << harris_keypoints->points.size() << endl;
    cout << "  Processing time: " << harris_duration.count() << " ms" << endl;
    cout << "  Keypoint ratio: " << static_cast<double>(harris_keypoints->size()) / cloud->size() * 100 << "%" << endl;

    // 获取关键点索引并转换为PointXYZ格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointIndicesConstPtr keypoints_indices = harris.getKeypointsIndices();
    pcl::copyPointCloud(*cloud, *keypoints_indices, *keypoints);

    cout << "  Converted to PointXYZ: " << keypoints->size() << " points" << endl;

    // 显示参数设置
    cout << "Harris Parameters:" << endl;
    cout << "  Method: LOWE" << endl;
    cout << "  Radius: 0.02" << endl;
    cout << "  Radius Search: 0.01" << endl;
    cout << "  Non-Max Supression: true" << endl;
    cout << "  Threshold: 0.002" << endl;
    cout << "  Refine: true" << endl;
    cout << "  Threads: 6" << endl;

    // 保存结果
    if (keypoints->size() > 0)
    {
        cout << "Saving keypoints to keypointsrgb.pcd..." << endl;
        pcl::io::savePCDFile("keypointsrgb.pcd", *keypoints, true);
        cout << "Keypoints saved successfully." << endl;
    }

    // 分析总结
    auto total_end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);

    cout << "Analysis Summary:" << endl;
    cout << "  Loading time: " << load_duration.count() << " ms" << endl;
    cout << "  Harris computation time: " << harris_duration.count() << " ms" << endl;
    cout << "  Total time: " << total_duration.count() << " ms" << endl;
    cout << "  Original points: " << cloud->size() << endl;
    cout << "  Keypoints: " << keypoints->size() << endl;
    cout << "  Keypoint ratio: " << static_cast<double>(keypoints->size()) / cloud->size() * 100 << "%" << endl;

    // 调用可视化函数
    visualize_harris_keypoints(cloud, keypoints, harris_duration.count(),
                               "Harris 3D Keypoint Extraction - ROPS Cloud Dataset");

    return 0;
}