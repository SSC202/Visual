#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <thread>
#include <chrono>

int main()
{
    // 创建点云指针
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);          // 原始点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>); // 下采样后点云

    // 读取点云数据
    std::cout << "=== Loading Point Cloud ===" << std::endl;
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("test.pcd", *cloud) == -1)
    {
        PCL_ERROR("Could not read point cloud file!\n");
        return -1;
    }
    std::cout << "Original point cloud: " << cloud->width * cloud->height
              << " points" << std::endl;

    // 体素网格下采样
    std::cout << "\n=== Performing Voxel Grid Downsampling ===" << std::endl;

    pcl::VoxelGrid<pcl::PointXYZ> vg;    // 创建体素网格滤波器对象
    vg.setInputCloud(cloud);             // 设置输入点云
    vg.setLeafSize(0.05f, 0.05f, 0.05f); // 设置体素尺寸
    vg.filter(*cloud_filtered);          // 执行下采样

    std::cout << "Downsampled point cloud: " << cloud_filtered->width * cloud_filtered->height
              << " points" << std::endl;
    std::cout << "Reduction ratio: "
              << (1.0 - static_cast<float>(cloud_filtered->size()) / cloud->size()) * 100
              << "%" << std::endl;
    std::cout << "Voxel leaf size: "
              << vg.getLeafSize()[0] << " x "
              << vg.getLeafSize()[1] << " x "
              << vg.getLeafSize()[2] << std::endl;

    // 保存下采样点云
    std::cout << "\n=== Saving Downsampled Point Cloud ===" << std::endl;

    if (cloud_filtered->empty())
    {
        PCL_ERROR("Downsampled point cloud is empty!\n");
        return -1;
    }
    else
    {
        pcl::io::savePCDFileASCII("filtered.pcd", *cloud_filtered);
        std::cout << "Downsampled point cloud saved to 'filtered.pcd'" << std::endl;
    }

    // 可视化

    // 创建可视化器
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Voxel Grid Downsampling Comparison"));
    viewer->setBackgroundColor(0.05, 0.05, 0.15); // 深蓝色背景

    // 左侧视口 - 原始点云
    int v1(0);
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->setBackgroundColor(0.1, 0.1, 0.2, v1);

    // 添加原始点云（红色）
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_red(cloud, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_red, "original_cloud", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud", v1);

    // 添加文本说明
    viewer->addText("Original Point Cloud", 10, 20, 16, 1, 1, 1, "original_text", v1);
    std::string original_count = "Points: " + std::to_string(cloud->size());
    viewer->addText(original_count, 10, 40, 14, 1, 1, 1, "original_count", v1);
    viewer->addText("(High density)", 10, 60, 14, 1, 1, 1, "density_note", v1);

    // 右侧视口 - 下采样点云
    int v2(0);
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viewer->setBackgroundColor(0.1, 0.2, 0.1, v2);

    // 添加下采样点云（绿色）
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> filtered_green(cloud_filtered, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_filtered, filtered_green, "downsampled_cloud", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "downsampled_cloud", v2);

    // 添加文本说明
    viewer->addText("Downsampled Point Cloud", 10, 20, 16, 1, 1, 1, "downsampled_text", v2);
    std::string downsampled_count = "Points: " + std::to_string(cloud_filtered->size());
    viewer->addText(downsampled_count, 10, 40, 14, 1, 1, 1, "downsampled_count", v2);

    // 添加下采样参数信息
    std::string voxel_info = "Voxel size: 0.05 x 0.05 x 0.05";
    viewer->addText(voxel_info, 10, 60, 14, 1, 1, 1, "voxel_info", v2);

    std::string reduction_info = "Reduction: " +
                                 std::to_string(static_cast<int>((1.0 - static_cast<float>(cloud_filtered->size()) / cloud->size()) * 100)) + "%";
    viewer->addText(reduction_info, 10, 80, 14, 1, 1, 1, "reduction_info", v2);

    // 公共设置

    // 添加标题
    viewer->addText("Voxel Grid Downsampling", 300, 20, 18, 1, 1, 1, "title");

    // 添加坐标轴
    viewer->addCoordinateSystem(1.0, "axis_v1", v1);
    viewer->addCoordinateSystem(1.0, "axis_v2", v2);

    // 设置相机参数
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 5, 0, 0, 0, 0, 1, 0);

    std::cout << "\n=== Visualization Started ===" << std::endl;
    std::cout << "Left: Original point cloud (Red) - high density" << std::endl;
    std::cout << "Right: Downsampled point cloud (Green) - reduced density" << std::endl;
    std::cout << "Voxel size: 0.05 x 0.05 x 0.05" << std::endl;
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