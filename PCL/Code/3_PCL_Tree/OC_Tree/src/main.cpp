#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/octree/octree_search.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/colors.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <string>

int main()
{
    // 创建点云数据 
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->width = 1000;
    cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);

    // 生成随机点云，集中在中心区域
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        cloud->points[i].x = 1024.0f * rand() / (RAND_MAX + 1.0f);
        cloud->points[i].y = 1024.0f * rand() / (RAND_MAX + 1.0f);
        cloud->points[i].z = 1024.0f * rand() / (RAND_MAX + 1.0f);
    }

    // 创建八叉树
    float resolution = 64.0f; // 八叉树体素分辨率
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);

    std::cout << "=== Building Octree ===" << std::endl;
    std::cout << "Resolution: " << resolution << std::endl;

    octree.setInputCloud(cloud);
    octree.addPointsFromInputCloud();

    std::cout << "Octree depth: " << octree.getTreeDepth() << std::endl;
    std::cout << "Leaf count: " << octree.getLeafCount() << std::endl;

    // 设置查询点
    pcl::PointXYZ searchPoint;
    searchPoint.x = 512.0f;
    searchPoint.y = 512.0f;
    searchPoint.z = 512.0f;

    /**
     * @brief   KNN 搜索
     */
    int k = 10;
    std::vector<int> point_idx_knn_search;
    std::vector<float> point_knn_squared_distance;

    std::cout << "\n=== Octree KNN Search ===" << std::endl;
    std::cout << "Search point: (" << searchPoint.x << ", " << searchPoint.y << ", " << searchPoint.z << ")" << std::endl;
    std::cout << "K = " << k << std::endl;

    if (octree.nearestKSearch(searchPoint, k, point_idx_knn_search, point_knn_squared_distance) > 0)
    {
        std::cout << "Found " << point_idx_knn_search.size() << " nearest neighbors:" << std::endl;
        for (size_t i = 0; i < point_idx_knn_search.size(); ++i)
        {
            std::cout << "  Point " << i + 1 << ": ("
                      << cloud->points[point_idx_knn_search[i]].x << ", "
                      << cloud->points[point_idx_knn_search[i]].y << ", "
                      << cloud->points[point_idx_knn_search[i]].z
                      << ") - distance: " << sqrt(point_knn_squared_distance[i]) << std::endl;
        }
    }

    /**
     * @brief   半径搜索
     */
    float radius = 200.0f;
    std::vector<int> point_idx_radius_search;
    std::vector<float> point_radius_squared_distance;

    std::cout << "\n=== Octree Radius Search ===" << std::endl;
    std::cout << "Search radius: " << radius << std::endl;

    if (octree.radiusSearch(searchPoint, radius, point_idx_radius_search, point_radius_squared_distance) > 0)
    {
        std::cout << "Found " << point_idx_radius_search.size() << " points within radius " << radius << std::endl;

        int display_count = std::min(5, (int)point_idx_radius_search.size());
        for (int i = 0; i < display_count; ++i)
        {
            std::cout << "  Point " << i + 1 << ": ("
                      << cloud->points[point_idx_radius_search[i]].x << ", "
                      << cloud->points[point_idx_radius_search[i]].y << ", "
                      << cloud->points[point_idx_radius_search[i]].z
                      << ") - distance: " << sqrt(point_radius_squared_distance[i]) << std::endl;
        }
        if (point_idx_radius_search.size() > 5)
        {
            std::cout << "  ... and " << point_idx_radius_search.size() - 5 << " more points" << std::endl;
        }
    }

    /**
     * @brief   体素搜索
     */
    std::cout << "\n=== Octree Voxel Search ===" << std::endl;
    std::vector<int> point_idx_voxel_search;

    if (octree.voxelSearch(searchPoint, point_idx_voxel_search))
    {
        std::cout << "Found " << point_idx_voxel_search.size() << " points in the same voxel:" << std::endl;
        for (size_t i = 0; i < point_idx_voxel_search.size(); ++i)
        {
            std::cout << "  Point " << i + 1 << ": ("
                      << cloud->points[point_idx_voxel_search[i]].x << ", "
                      << cloud->points[point_idx_voxel_search[i]].y << ", "
                      << cloud->points[point_idx_voxel_search[i]].z << ")" << std::endl;
        }
    }

    /**
     * @brief   可视化
     */

    // 创建可视化器
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Octree Search Visualization"));
    viewer->setBackgroundColor(0.05, 0.05, 0.15);

    // 添加原始点云
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_gray(cloud, 180, 180, 180);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_gray, "original_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud");

    // 添加查询点
    pcl::PointCloud<pcl::PointXYZ>::Ptr search_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    search_cloud->push_back(searchPoint);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> search_red(search_cloud, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(search_cloud, search_red, "search_point");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 20, "search_point");

    // 添加 KNN 邻居点
    pcl::PointCloud<pcl::PointXYZ>::Ptr knn_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < point_idx_knn_search.size(); ++i)
    {
        knn_cloud->push_back(cloud->points[point_idx_knn_search[i]]);
    }
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> knn_green(knn_cloud, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(knn_cloud, knn_green, "knn_points");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "knn_points");

    // 添加半径搜索点
    pcl::PointCloud<pcl::PointXYZ>::Ptr radius_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < point_idx_radius_search.size(); ++i)
    {
        radius_cloud->push_back(cloud->points[point_idx_radius_search[i]]);
    }
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> radius_blue(radius_cloud, 100, 150, 255);
    viewer->addPointCloud<pcl::PointXYZ>(radius_cloud, radius_blue, "radius_points");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "radius_points");

    // 添加体素内点
    pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < point_idx_voxel_search.size(); ++i)
    {
        voxel_cloud->push_back(cloud->points[point_idx_voxel_search[i]]);
    }
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> voxel_yellow(voxel_cloud, 255, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(voxel_cloud, voxel_yellow, "voxel_points");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "voxel_points");

    // 添加搜索半径球体
    viewer->addSphere(searchPoint, radius, 0.3, 0.3, 0.8, "radius_sphere");
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.15, "radius_sphere");

    // 添加八叉树体素边界框
    std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> voxel_centers;
    octree.getOccupiedVoxelCenters(voxel_centers);

    int voxel_count = 0;
    for (const auto &center : voxel_centers)
    {
        // 只显示部分体素避免过于密集
        if (voxel_count++ % 5 == 0)
        {
            std::string voxel_id = "voxel_" + std::to_string(voxel_count);
            viewer->addCube(center.x - resolution / 2, center.x + resolution / 2,
                            center.y - resolution / 2, center.y + resolution / 2,
                            center.z - resolution / 2, center.z + resolution / 2,
                            0.8, 0.8, 0.8, voxel_id);
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.1, voxel_id);
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                                                pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, voxel_id);
        }
    }

    // 添加图例文本
    viewer->addText("Octree Search Visualization", 10, 150, 18, 1, 1, 1, "title");
    viewer->addText("Gray: Original point cloud", 10, 130, 14, 0.7, 0.7, 0.7, "original_text");
    viewer->addText("Red: Search point", 10, 115, 14, 1, 0, 0, "search_text");
    viewer->addText("Green: KNN neighbors (K=10)", 10, 100, 14, 0, 1, 0, "knn_text");
    viewer->addText("Blue: Radius search points", 10, 85, 14, 0.4, 0.6, 1, "radius_text");
    viewer->addText("Yellow: Same voxel points", 10, 70, 14, 1, 1, 0, "voxel_text");
    viewer->addText("Wireframe cubes: Octree voxels", 10, 55, 14, 0.8, 0.8, 0.8, "voxel_structure_text");

    // 添加统计信息
    std::string stats_text = "Results: KNN=" + std::to_string(point_idx_knn_search.size()) +
                             ", Radius=" + std::to_string(point_idx_radius_search.size()) +
                             ", Voxel=" + std::to_string(point_idx_voxel_search.size());
    viewer->addText(stats_text, 10, 170, 16, 1, 1, 1, "stats_text");

    std::string octree_info = "Octree: Res=" + std::to_string(resolution) +
                              ", Depth=" + std::to_string(octree.getTreeDepth()) +
                              ", Leaves=" + std::to_string(octree.getLeafCount());
    viewer->addText(octree_info, 10, 185, 14, 0.8, 0.8, 1, "octree_info");

    // 添加坐标轴
    viewer->addCoordinateSystem(200.0);

    // 设置相机位置
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, -800, 800, 0, 0, 400, 0, 0, 1);

    std::cout << "\n=== Visualization Started ===" << std::endl;
    std::cout << "Press 'q' in the window to exit" << std::endl;
    std::cout << "Use mouse to rotate and scroll to zoom" << std::endl;
    std::cout << "Press 'r' to reset camera view" << std::endl;

    // 主循环
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}