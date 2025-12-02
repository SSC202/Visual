#include <iostream>
#include <thread>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/keypoints/susan.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>

using namespace std;

/**
 * @brief Visualize SUSAN keypoint extraction results using dual viewports
 * 使用双视口可视化SUSAN关键点提取结果
 */
void visualize_susan_keypoints(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr &keypoints,
                               double processing_time,
                               const std::string &window_title = "SUSAN Keypoint Extraction")
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
    std::string file_info = "File: cloud.pcd";
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

    viewer->addText("SUSAN Keypoint Extraction", 10, 20, 16, 1, 1, 1, "keypoints_text", v2);
    viewer->addText("Red: Extracted Keypoints", 10, 40, 14, 1, 0, 0, "keypoints_red_text", v2);

    std::string keypoints_info = "Keypoints: " + std::to_string(keypoints->size());
    std::string keypoints_ratio = "Keypoint Ratio: " +
                                  std::to_string(static_cast<double>(keypoints->size()) / cloud->size() * 100).substr(0, 5) + "%";
    std::string time_info = "Processing Time: " + std::to_string(processing_time) + " ms";
    viewer->addText(keypoints_info, 10, 60, 12, 1, 1, 1, "keypoints_info", v2);
    viewer->addText(keypoints_ratio, 10, 75, 12, 1, 1, 1, "keypoints_ratio", v2);
    viewer->addText(time_info, 10, 90, 12, 1, 1, 1, "time_info", v2);

    viewer->addCoordinateSystem(0.1, "axis_v2", v2);

    // 设置相机参数
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 5, 0, 0, 0, 0, 1, 0);

    // 主循环
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main()
{
    cout << "SUSAN Keypoint Extraction Analysis" << endl;
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
    cout << "  Loaded " << cloud->size() << " points." << endl;
    cout << "  Loading time: " << load_duration.count() << " ms" << endl;

    // SUSAN关键点提取
    cout << "Computing SUSAN Keypoints:" << endl;
    auto susan_start_time = std::chrono::steady_clock::now();

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::SUSANKeypoint<pcl::PointXYZ, pcl::PointXYZI> susan;

    susan.setInputCloud(cloud);         // 设置输入点云
    susan.setSearchMethod(tree);        // 设置邻域搜索的方式
    susan.setNumberOfThreads(12);       // 设置多线程加速的线程数
    susan.setRadius(3.0f);              // 设置法向量估计和非极大值抑制的半径
    susan.setDistanceThreshold(0.001f); // 设置距离阈值
    susan.setAngularThreshold(0.0001f); // 设置用于角点检测的角度阈值
    susan.setIntensityThreshold(0.1f);  // 设置用于角点检测的强度阈值
    susan.setNonMaxSupression(true);    // 对响应应用非最大值抑制，以保持最强角

    pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints_xyzi(new pcl::PointCloud<pcl::PointXYZI>());
    susan.compute(*keypoints_xyzi);

    // 带强度的点云PointXYZI转换成PointXYZ
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*keypoints_xyzi, *keypoints_xyz);

    auto susan_end_time = std::chrono::steady_clock::now();
    auto susan_duration = std::chrono::duration_cast<std::chrono::milliseconds>(susan_end_time - susan_start_time);

    cout << "  SUSAN keypoints extracted: " << keypoints_xyz->points.size() << endl;
    cout << "  Processing time: " << susan_duration.count() << " ms" << endl;
    cout << "  Keypoint ratio: " << static_cast<double>(keypoints_xyz->size()) / cloud->size() * 100 << "%" << endl;

    // 显示参数设置
    cout << "SUSAN Parameters:" << endl;
    cout << "  Radius: 3.0" << endl;
    cout << "  Distance threshold: 0.001" << endl;
    cout << "  Angular threshold: 0.0001" << endl;
    cout << "  Intensity threshold: 0.1" << endl;
    cout << "  Non-maximum suppression: Enabled" << endl;
    cout << "  Threads: 6" << endl;

    // 分析总结
    auto total_end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);

    cout << "Analysis Summary:" << endl;
    cout << "  Loading time: " << load_duration.count() << " ms" << endl;
    cout << "  SUSAN computation time: " << susan_duration.count() << " ms" << endl;
    cout << "  Total time: " << total_duration.count() << " ms" << endl;
    cout << "  Original points: " << cloud->size() << endl;
    cout << "  Keypoints: " << keypoints_xyz->size() << endl;
    cout << "  Keypoint ratio: " << static_cast<double>(keypoints_xyz->size()) / cloud->size() * 100 << "%" << endl;

    // 调用可视化函数
    visualize_susan_keypoints(cloud, keypoints_xyz, susan_duration.count(), "SUSAN Keypoint Extraction - Pig Dataset");

    return 0;
}