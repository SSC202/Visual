#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <random>

int main(int argc, char **argv)
{
    // 读取点云数据
    std::cout << "=== Loading Point Cloud ===" << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>("test.pcd", *cloud) == -1)
    {
        PCL_ERROR("Could not read point cloud file!\n");
        return -1;
    }
    std::cout << "Original point cloud: " << cloud->points.size() << " data points" << std::endl;

    // 体素网格下采样
    std::cout << "\n=== Performing Voxel Grid Downsampling ===" << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.01f, 0.01f, 0.01f); // 1cm 体素尺寸
    vg.filter(*cloud_filtered);

    std::cout << "Downsampled point cloud: " << cloud_filtered->points.size() << " data points" << std::endl;
    std::cout << "Reduction ratio: "
              << (1.0 - static_cast<float>(cloud_filtered->size()) / cloud->size()) * 100
              << "%" << std::endl;

    // 平面分割
    std::cout << "\n=== Performing Plane Segmentation ===" << std::endl;

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZ>);

    // 设置分割参数
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.02);

    int i = 0;
    int nr_points = static_cast<int>(cloud_filtered->points.size());

    // 迭代移除平面，直到剩余点云小于原始点云的30%
    while (cloud_filtered->points.size() > 0.3 * nr_points)
    {
        std::cout << "-> Segmenting plane " << i + 1 << std::endl;

        seg.setInputCloud(cloud_filtered);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.size() == 0)
        {
            std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
            break;
        }

        // 提取平面内点
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_filtered);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*cloud_plane);

        std::cout << "  Planar component " << i + 1 << ": " << cloud_plane->points.size() << " data points" << std::endl;

        // 移除平面内点，保留剩余点云
        extract.setNegative(true);
        extract.filter(*cloud_f);
        *cloud_filtered = *cloud_f;

        i++;
    }

    std::cout << "Remaining non-planar points: " << cloud_filtered->points.size() << std::endl;

    // 欧几里得聚类
    std::cout << "\n=== Performing Euclidean Clustering ===" << std::endl;

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

    ec.setClusterTolerance(0.02); // 2cm 聚类容差
    ec.setMinClusterSize(100);    // 最小聚类点数
    ec.setMaxClusterSize(25000);  // 最大聚类点数
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_filtered);
    ec.extract(cluster_indices);

    std::cout << "Found " << cluster_indices.size() << " clusters" << std::endl;

    // 可视化

    // 创建可视化器
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Cloud Segmentation and Clustering"));
    viewer->setBackgroundColor(0.05, 0.05, 0.15); // 深蓝色背景

    // 左侧视口 - 原始点云
    int v1(0);
    viewer->createViewPort(0.0, 0.0, 0.33, 1.0, v1);
    viewer->setBackgroundColor(0.1, 0.1, 0.2, v1);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_white(cloud, 255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_white, "original_cloud", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original_cloud", v1);

    viewer->addText("Original Point Cloud", 10, 20, 14, 1, 1, 1, "original_text", v1);
    std::string original_count = "Points: " + std::to_string(cloud->size());
    viewer->addText(original_count, 10, 40, 12, 1, 1, 1, "original_count", v1);

    // 中间视口 - 下采样和平面移除后点云
    int v2(0);
    viewer->createViewPort(0.33, 0.0, 0.66, 1.0, v2);
    viewer->setBackgroundColor(0.1, 0.15, 0.1, v2);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> filtered_yellow(cloud_filtered, 255, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_filtered, filtered_yellow, "filtered_cloud", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "filtered_cloud", v2);

    viewer->addText("After Downsampling & Plane Removal", 10, 20, 14, 1, 1, 1, "filtered_text", v2);
    std::string filtered_count = "Points: " + std::to_string(cloud_filtered->size());
    viewer->addText(filtered_count, 10, 40, 12, 1, 1, 1, "filtered_count", v2);
    std::string planes_removed = "Planes removed: " + std::to_string(i);
    viewer->addText(planes_removed, 10, 60, 12, 1, 1, 1, "planes_info", v2);

    // 右侧视口 - 聚类结果
    int v3(0);
    viewer->createViewPort(0.66, 0.0, 1.0, 1.0, v3);
    viewer->setBackgroundColor(0.15, 0.1, 0.1, v3);

    // 为每个聚类添加不同颜色的点云
    std::vector<pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>> color_handlers;
    std::vector<std::vector<uint8_t>> colors = {
        {255, 0, 0},
        {0, 255, 0},
        {0, 0, 255},
        {255, 255, 0},
        {255, 0, 255},
        {0, 255, 255},
        {255, 128, 0},
        {128, 0, 255}};

    int j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin();
         it != cluster_indices.end(); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);

        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
            cloud_cluster->points.push_back(cloud_filtered->points[*pit]);

        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        std::cout << "Cluster " << j << ": " << cloud_cluster->points.size() << " data points" << std::endl;

        // 为聚类分配颜色
        std::vector<uint8_t> color = colors[j % colors.size()];
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cluster_color(
            cloud_cluster, color[0], color[1], color[2]);

        std::string cluster_id = "cluster_" + std::to_string(j);
        viewer->addPointCloud<pcl::PointXYZ>(cloud_cluster, cluster_color, cluster_id, v3);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cluster_id, v3);

        j++;
    }

    viewer->addText("Clustering Results", 10, 20, 14, 1, 1, 1, "clusters_text", v3);
    std::string clusters_count = "Clusters: " + std::to_string(cluster_indices.size());
    viewer->addText(clusters_count, 10, 40, 12, 1, 1, 1, "clusters_count", v3);

    // 公共设置

    // 添加标题
    viewer->addText("Point Cloud Segmentation and Clustering", 400, 20, 16, 1, 1, 1, "title");

    // 添加坐标轴
    viewer->addCoordinateSystem(0.5, "axis_v1", v1);
    viewer->addCoordinateSystem(0.5, "axis_v2", v2);
    viewer->addCoordinateSystem(0.5, "axis_v3", v3);

    // 设置相机参数
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 5, 0, 0, 0, 0, 1, 0);

    // 添加参数信息
    std::string params_text = "Parameters: Voxel=0.01, PlaneDistThresh=0.02, ClusterTol=0.02";
    viewer->addText(params_text, 10, 450, 12, 1, 1, 1, "params_text");

    std::cout << "\n=== Visualization Started ===" << std::endl;
    std::cout << "Left: Original point cloud (White)" << std::endl;
    std::cout << "Middle: After downsampling & plane removal (Yellow)" << std::endl;
    std::cout << "Right: Clustering results (Colored clusters)" << std::endl;
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