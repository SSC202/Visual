#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <thread>
#include <chrono>

int main()
{
    // 创建点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->width = 1000;
    cloud->height = 1; // 无序点云
    cloud->points.resize(cloud->width * cloud->height);

    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        cloud->points[i].x = 1024.0f * rand() / (RAND_MAX + 1.0f);
        cloud->points[i].y = 1024.0f * rand() / (RAND_MAX + 1.0f);
        cloud->points[i].z = 1024.0f * rand() / (RAND_MAX + 1.0f);
    }

    // 创建 KD 树
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    // KNN 搜索
    pcl::PointXYZ searchPoint;
    searchPoint.x = 1024.0f * rand() / (RAND_MAX + 1.0f);
    searchPoint.y = 1024.0f * rand() / (RAND_MAX + 1.0f);
    searchPoint.z = 1024.0f * rand() / (RAND_MAX + 1.0f);

    int k = 5;                                        // 查询 5 个最近邻点
    std::vector<int> point_idx_knn_search(k);         // 存储 K 近邻索引
    std::vector<float> point_knn_squared_distance(k); // 存储 K 近邻的平方距离

    std::cout << "=== KNN Search Results ===" << std::endl;
    std::cout << "Search point: (" << searchPoint.x << ", " << searchPoint.y << ", " << searchPoint.z << ")" << std::endl;
    std::cout << "K = " << k << std::endl;

    if (kdtree.nearestKSearch(searchPoint, k, point_idx_knn_search, point_knn_squared_distance) > 0)
    {
        for (size_t i = 0; i < point_idx_knn_search.size(); ++i)
        {
            std::cout << "  " << cloud->points[point_idx_knn_search[i]].x << " " << cloud->points[point_idx_knn_search[i]].y << " " << cloud->points[point_idx_knn_search[i]].z << " (squared distance: " << point_knn_squared_distance[i] << ")" << std::endl;
        }
    }

    // 半径搜索
    float radius = 300.0f; // 搜索半径
    std::vector<int> point_idx_radius_search;
    std::vector<float> point_radius_squared_distance;

    std::cout << "\n=== Radius Search Results ===" << std::endl;
    std::cout << "Search radius: " << radius << std::endl;

    if (kdtree.radiusSearch(searchPoint, radius, point_idx_radius_search, point_radius_squared_distance) > 0)
    {
        std::cout << "Found " << point_idx_radius_search.size() << " points within radius " << radius << std::endl;

        // 显示前10个点
        int display_count = std::min(10, (int)point_idx_radius_search.size());
        for (int i = 0; i < display_count; ++i)
        {
            std::cout << "  Point " << i + 1 << ": (" << cloud->points[point_idx_radius_search[i]].x << ", " << cloud->points[point_idx_radius_search[i]].y << ", " << cloud->points[point_idx_radius_search[i]].z << ") - squared distance: " << point_radius_squared_distance[i] << std::endl;
        }
        if (point_idx_radius_search.size() > 10)
        {
            std::cout << "  ... and " << point_idx_radius_search.size() - 10 << " more points" << std::endl;
        }
    }

    // 可视化
    // 创建可视化器
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("KD-Tree Search Visualization"));
    viewer->setBackgroundColor(0.05, 0.05, 0.15); // 深蓝色背景

    // 添加原始点云
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_gray(cloud, 200, 200, 200);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_gray, "original_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud");

    // 创建查询点点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr search_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    search_cloud->push_back(searchPoint);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> search_red(search_cloud, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(search_cloud, search_red, "search_point");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "search_point");

    // 创建 KNN 点点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr knn_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < point_idx_knn_search.size(); ++i)
    {
        knn_cloud->push_back(cloud->points[point_idx_knn_search[i]]);
    }
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> knn_green(knn_cloud, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(knn_cloud, knn_green, "knn_points");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "knn_points");

    // 创建半径搜索点点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr radius_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < point_idx_radius_search.size(); ++i)
    {
        radius_cloud->push_back(cloud->points[point_idx_radius_search[i]]);
    }
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> radius_blue(radius_cloud, 0, 100, 255);
    viewer->addPointCloud<pcl::PointXYZ>(radius_cloud, radius_blue, "radius_points");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "radius_points");

    // 添加从查询点到每个KNN点的连线
    for (size_t i = 0; i < point_idx_knn_search.size(); ++i)
    {
        pcl::PointXYZ neighbor_point = cloud->points[point_idx_knn_search[i]];
        std::string line_id = "knn_line_" + std::to_string(i);
        viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(searchPoint, neighbor_point, 255, 255, 0, line_id);
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, line_id);
    }

    // 添加搜索半径球体
    viewer->addSphere(searchPoint, radius, 0.3, 0.3, 0.8, "radius_sphere");
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "radius_sphere");
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                                        pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "radius_sphere");

    // 添加图例文本
    viewer->addText("KD-Tree Search Visualization", 10, 110, 20, 1, 1, 1, "title");
    viewer->addText("Gray: Original point cloud (1000 points)", 10, 90, 14, 0.8, 0.8, 0.8, "original_text");
    viewer->addText("Red: Search point", 10, 75, 14, 1, 0, 0, "search_text");
    viewer->addText("Green: KNN neighbors (K=5)", 10, 60, 14, 0, 1, 0, "knn_text");
    viewer->addText("Blue: Radius search points", 10, 45, 14, 0, 0.6, 1, "radius_text");
    viewer->addText("Yellow: KNN connection lines", 10, 30, 14, 1, 1, 0, "lines_text");
    viewer->addText("Transparent sphere: Search radius", 10, 15, 14, 0.3, 0.3, 0.8, "sphere_text");

    // 添加统计信息
    std::string stats_text = "Results: KNN=" + std::to_string(point_idx_knn_search.size()) +
                             ", Radius=" + std::to_string(point_idx_radius_search.size());
    viewer->addText(stats_text, 10, 130, 16, 1, 1, 1, "stats_text");

    // 添加坐标轴
    viewer->addCoordinateSystem(200.0);

    // 设置相机位置以获得更好的视角
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 2500, 0, 0, 0, 0, 1, 0);

    std::cout << "\n=== Visualization Started ===" << std::endl;
    std::cout << "Press 'q' in the window to exit" << std::endl;
    std::cout << "Use mouse to rotate and scroll to zoom" << std::endl;
    std::cout << "Press 'r' to reset camera view" << std::endl;

    // 主循环 - 保持可视化窗口打开
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}