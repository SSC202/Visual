#include <iostream>
#include <string>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/time.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>
#include <thread>
#include <chrono>

#include "dbscan.h"

int main()
{
    // 读取点云数据
    std::cout << "=== Loading Point Cloud ===" << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>("test.pcd", *cloud) == -1)
    {
        PCL_ERROR("Could not read point cloud file!\n");
        return -1;
    }
    std::cout << "Point cloud loaded: " << cloud->points.size() << " points" << std::endl;

    // DBSCAN 密度聚类
    std::cout << "\n=== Performing DBSCAN Clustering ===" << std::endl;

    pcl::StopWatch timer;
    std::vector<pcl::Indices> cluster_indices;

    double epsilon = 1.0; // 邻域距离阈值 (米)
    int min_points = 50;  // 最小点数

    bool success = dbscan(*cloud, cluster_indices, epsilon, min_points);

    if (!success)
    {
        PCL_ERROR("DBSCAN clustering failed!\n");
        return -1;
    }

    std::cout << "DBSCAN clustering completed in " << timer.getTimeSeconds() << " seconds" << std::endl;
    std::cout << "Found " << cluster_indices.size() << " clusters" << std::endl;

    // 聚类结果处理和保存
    std::cout << "\n=== Processing and Saving Clustering Results ===" << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::vector<std::vector<uint8_t>> colors = {
        {255, 0, 0},   // 红色
        {0, 255, 0},   // 绿色
        {0, 0, 255},   // 蓝色
        {255, 255, 0}, // 黄色
        {255, 0, 255}, // 紫色
        {0, 255, 255}, // 青色
        {255, 128, 0}, // 橙色
        {128, 0, 255}, // 紫罗兰色
        {128, 255, 0}, // 黄绿色
        {0, 128, 255}  // 天蓝色
    };

    int cluster_id = 1;
    for (const auto &cluster : cluster_indices)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        // 为每个聚类分配颜色
        std::vector<uint8_t> color = colors[(cluster_id - 1) % colors.size()];

        // 填充聚类点云
        for (const auto &point_index : cluster)
        {
            pcl::PointXYZRGB colored_point;
            colored_point.x = cloud->points[point_index].x;
            colored_point.y = cloud->points[point_index].y;
            colored_point.z = cloud->points[point_index].z;
            colored_point.r = color[0];
            colored_point.g = color[1];
            colored_point.b = color[2];
            cluster_cloud->points.push_back(colored_point);
        }

        cluster_cloud->width = cluster_cloud->points.size();
        cluster_cloud->height = 1;
        cluster_cloud->is_dense = true;

        std::cout << "Cluster " << cluster_id << ": " << cluster_cloud->points.size() << " points" << std::endl;

        // 保存单个聚类
        std::string filename = "dbscan_cluster_" + std::to_string(cluster_id) + ".pcd";
        pcl::io::savePCDFileASCII(filename, *cluster_cloud);

        // 合并到总点云用于可视化
        *clustered_cloud += *cluster_cloud;
        cluster_id++;
    }

    std::cout << "Clusters saved to individual PCD files" << std::endl;

    // 可视化

    // 创建可视化器
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("DBSCAN Clustering Results"));
    viewer->setBackgroundColor(0.05, 0.05, 0.15); // 深蓝色背景

    // 左侧视口 - 原始点云
    int v1(0);
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->setBackgroundColor(0.1, 0.1, 0.2, v1);

    // 添加原始点云（白色）
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_white(cloud, 255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_white, "original_cloud", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud", v1);

    // 添加文本说明
    viewer->addText("Original Point Cloud", 10, 20, 16, 1, 1, 1, "original_text", v1);
    std::string original_count = "Points: " + std::to_string(cloud->size());
    viewer->addText(original_count, 10, 40, 14, 1, 1, 1, "original_count", v1);

    // 右侧视口 - 聚类结果
    int v2(0);
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viewer->setBackgroundColor(0.1, 0.15, 0.1, v2);

    // 添加聚类结果点云
    viewer->addPointCloud<pcl::PointXYZRGB>(clustered_cloud, "clustered_cloud", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "clustered_cloud", v2);

    // 添加文本说明
    viewer->addText("DBSCAN Clustering Results", 10, 20, 16, 1, 1, 1, "clustered_text", v2);
    std::string cluster_count = "Clusters: " + std::to_string(cluster_indices.size());
    viewer->addText(cluster_count, 10, 40, 14, 1, 1, 1, "cluster_count", v2);

    // 添加参数信息
    std::string params_info = "Epsilon: " + std::to_string(epsilon) + ", MinPts: " + std::to_string(min_points);
    viewer->addText(params_info, 10, 60, 14, 1, 1, 1, "params_info", v2);

    std::string time_info = "Time: " + std::to_string(timer.getTimeSeconds()) + "s";
    viewer->addText(time_info, 10, 80, 14, 1, 1, 1, "time_info", v2);

    // 公共设置

    // 添加标题
    viewer->addText("DBSCAN Density-Based Clustering", 300, 20, 18, 1, 1, 1, "title");

    // 添加坐标轴
    viewer->addCoordinateSystem(1.0, "axis_v1", v1);
    viewer->addCoordinateSystem(1.0, "axis_v2", v2);

    // 设置相机参数
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 5, 0, 0, 0, 0, 1, 0);

    std::cout << "\n=== Visualization Started ===" << std::endl;
    std::cout << "Left: Original point cloud (White)" << std::endl;
    std::cout << "Right: DBSCAN clustering results (Colored clusters)" << std::endl;
    std::cout << "Parameters: Epsilon=" << epsilon << ", MinPoints=" << min_points << std::endl;
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