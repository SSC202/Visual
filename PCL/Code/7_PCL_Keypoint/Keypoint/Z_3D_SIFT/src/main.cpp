#include <iostream>
#include <thread>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>

using namespace std;

// 基于Z梯度估计3D点云的SIFT关键点
namespace pcl
{
    template <>
    struct SIFTKeypointFieldSelector<PointXYZ>
    {
        inline float
        operator()(const PointXYZ &p) const
        {
            return p.z;
        }
    };
}

/**
 * @brief Visualize SIFT keypoint extraction results
 * 可视化SIFT关键点提取结果
 */
void visualize_sift_keypoints(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr &keypoints,
                              double processing_time,
                              const std::string &window_title = "SIFT Keypoint Extraction")
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
    std::string file_info = "File: bunny.pcd";
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
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "keypoints_cloud", v2);

    viewer->addText("SIFT Keypoint Extraction", 10, 20, 16, 1, 1, 1, "keypoints_text", v2);
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
    viewer->setCameraPosition(0, 0, 0.3, 0, 0, 0, 0, 1, 0); // 适应兔子点云

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
    cout << "==================================" << endl;

    auto total_start_time = std::chrono::steady_clock::now();

    // 加载点云数据
    auto load_start_time = std::chrono::steady_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    string filename = "cloud.pcd";

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud_xyz) == -1)
    {
        cerr << "Error: Could not read PCD file: " << filename << endl;
        cerr << "Please ensure the file exists in the current directory." << endl;
        return -1;
    }

    auto load_end_time = std::chrono::steady_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end_time - load_start_time);

    cout << "Point Cloud Information:" << endl;
    cout << "  Loaded " << cloud_xyz->size() << " points from " << filename << endl;
    cout << "  Loading time: " << load_duration.count() << " ms" << endl;

    if (cloud_xyz->empty())
    {
        cerr << "Error: Loaded point cloud is empty!" << endl;
        return -1;
    }

    // SIFT算法参数
    const float min_scale = 0.05f;      // 设置尺度空间中最小尺度的标准偏差
    const int n_octaves = 3;            // 设置尺度空间层数，越小则特征点越多
    const int n_scales_per_octave = 15; // 设置尺度空间中计算的尺度个数
    const float min_contrast = 0.01f;  // 设置限制关键点检测的阈值

    // SIFT关键点检测
    cout << "Computing SIFT Keypoints:" << endl;
    auto sift_start_time = std::chrono::steady_clock::now();

    pcl::SIFTKeypoint<pcl::PointXYZ, pcl::PointWithScale> sift; // 创建sift关键点检测对象
    pcl::PointCloud<pcl::PointWithScale> result;
    sift.setInputCloud(cloud_xyz); // 设置输入点云
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    sift.setSearchMethod(tree);                                // 创建一个空的kd树对象tree，并把它传递给sift检测对象
    sift.setScales(min_scale, n_octaves, n_scales_per_octave); // 指定搜索关键点的尺度范围
    sift.setMinimumContrast(min_contrast);                     // 设置限制关键点检测的阈值
    sift.compute(result);                                      // 执行sift关键点检测，保存结果在result

    auto sift_end_time = std::chrono::steady_clock::now();
    auto sift_duration = std::chrono::duration_cast<std::chrono::milliseconds>(sift_end_time - sift_start_time);

    cout << "  SIFT keypoints extracted: " << result.size() << endl;
    cout << "  Processing time: " << sift_duration.count() << " ms" << endl;

    // 为了可视化需要将点类型pcl::PointWithScale的数据转换为点类型pcl::PointXYZ的数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(result, *keypoints);

    // 显示参数设置
    cout << "SIFT Parameters:" << endl;
    cout << "  Min scale: " << min_scale << endl;
    cout << "  Number of octaves: " << n_octaves << endl;
    cout << "  Scales per octave: " << n_scales_per_octave << endl;
    cout << "  Minimum contrast: " << min_contrast << endl;

    // 分析总结
    auto total_end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);

    cout << "Analysis Summary:" << endl;
    cout << "  Loading time: " << load_duration.count() << " ms" << endl;
    cout << "  SIFT computation time: " << sift_duration.count() << " ms" << endl;
    cout << "  Total time: " << total_duration.count() << " ms" << endl;
    cout << "  Original points: " << cloud_xyz->size() << endl;
    cout << "  Keypoints: " << keypoints->size() << endl;
    cout << "  Keypoint ratio: " << static_cast<double>(keypoints->size()) / cloud_xyz->size() * 100 << "%" << endl;

    // 调用可视化函数
    visualize_sift_keypoints(cloud_xyz, keypoints, sift_duration.count(), "SIFT Keypoint Extraction - Bunny Dataset");

    return 0;
}