#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <thread>
#include <chrono>

int main()
{
    // 创建点云指针
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);          // 原始点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>); // 滤波后点云

    // 读取点云数据
    std::cout << "=== Loading Point Cloud ===" << std::endl;
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("test.pcd", *cloud) == -1)
    {
        PCL_ERROR("Could not read point cloud file!\n");
        return -1;
    }
    std::cout << "Original point cloud: " << cloud->width * cloud->height
              << " points" << std::endl;

    // 统计离群点去除
    std::cout << "\n=== Performing Statistical Outlier Removal ===" << std::endl;

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor; // 创建统计滤波器对象
    sor.setInputCloud(cloud);                          // 设置输入点云
    sor.setMeanK(50);                                  // 设置每个点的近邻点数量
    sor.setStddevMulThresh(1.0);                       // 设置标准差倍数阈值

    sor.filter(*cloud_filtered); // 执行滤波

    std::cout << "Filtered point cloud: " << cloud_filtered->width * cloud_filtered->height
              << " points" << std::endl;
    std::cout << "Removed " << cloud->size() - cloud_filtered->size()
              << " outlier points" << std::endl;
    std::cout << "Parameters: MeanK=" << sor.getMeanK()
              << ", StddevMulThresh=" << sor.getStddevMulThresh() << std::endl;

    // 保存滤波结果
    std::cout << "\n=== Saving Filtered Point Cloud ===" << std::endl;

    if (cloud_filtered->empty())
    {
        PCL_ERROR("Filtered point cloud is empty!\n");
        return -1;
    }
    else
    {
        pcl::io::savePCDFileASCII("filtered.pcd", *cloud_filtered);
        std::cout << "Filtered point cloud saved to 'filtered.pcd'" << std::endl;
    }

    // 可视化

    // 创建可视化器
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Statistical Outlier Removal Comparison"));
    viewer->setBackgroundColor(0.05, 0.05, 0.15); // 深蓝色背景

    // 左侧视口 - 原始点云
    int v1(0);
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->setBackgroundColor(0.1, 0.1, 0.2, v1);

    // 添加原始点云
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_red(cloud, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_red, "original_cloud", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud", v1);

    // 添加文本说明
    viewer->addText("Original Point Cloud", 10, 20, 16, 1, 1, 1, "original_text", v1);
    std::string original_count = "Points: " + std::to_string(cloud->size());
    viewer->addText(original_count, 10, 40, 14, 1, 1, 1, "original_count", v1);
    viewer->addText("(May contain outliers)", 10, 60, 14, 1, 1, 1, "outlier_note", v1);

    // 右侧视口 - 滤波后点云
    int v2(0);
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viewer->setBackgroundColor(0.1, 0.2, 0.1, v2);

    // 添加滤波后点云
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> filtered_green(cloud_filtered, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_filtered, filtered_green, "filtered_cloud", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "filtered_cloud", v2);

    // 添加文本说明
    viewer->addText("Filtered Point Cloud", 10, 20, 16, 1, 1, 1, "filtered_text", v2);
    std::string filtered_count = "Points: " + std::to_string(cloud_filtered->size());
    viewer->addText(filtered_count, 10, 40, 14, 1, 1, 1, "filtered_count", v2);

    // 添加滤波参数信息
    std::string filter_params = "MeanK: " + std::to_string(sor.getMeanK());
    viewer->addText(filter_params, 10, 60, 14, 1, 1, 1, "meanK_info", v2);

    std::string threshold_info = "Stddev Threshold: " + std::to_string(sor.getStddevMulThresh());
    viewer->addText(threshold_info, 10, 80, 14, 1, 1, 1, "threshold_info", v2);

    std::string removed_info = "Outliers Removed: " + std::to_string(cloud->size() - cloud_filtered->size());
    viewer->addText(removed_info, 10, 100, 14, 1, 1, 1, "removed_info", v2);

    // 公共设置

    // 添加标题
    viewer->addText("Statistical Outlier Removal Filter", 300, 20, 18, 1, 1, 1, "title");

    // 添加坐标轴
    viewer->addCoordinateSystem(1.0, "axis_v1", v1);
    viewer->addCoordinateSystem(1.0, "axis_v2", v2);

    // 设置相机参数
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 5, 0, 0, 0, 0, 1, 0);

    std::cout << "\n=== Visualization Started ===" << std::endl;
    std::cout << "Left: Original point cloud (Red) - may contain outliers" << std::endl;
    std::cout << "Right: Filtered point cloud (Green) - outliers removed" << std::endl;
    std::cout << "Filter parameters: MeanK=50, StddevMulThresh=1.0" << std::endl;
    std::cout << "Press 'q' to exit" << std::endl;
    std::cout << "Use mouse to rotate and scroll to zoom" << std::endl;

    // 主循环
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}