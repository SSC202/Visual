#include <iostream>
#include <thread>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/boundary.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/time.h>

using namespace std;

/**
 * @brief Visualize boundary extraction results using dual viewports
 * 使用双视口可视化边界提取结果
 */
void visualize_boundary_extraction(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                   pcl::PointCloud<pcl::PointXYZ>::Ptr &boundary_points,
                                   const std::string &window_title = "Boundary Extraction Visualization")
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0.05, 0.05, 0.15); // 深蓝色背景

    // 创建两个视口
    int v1(0), v2(1);
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1); // 左：原始点云
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2); // 右：边界点云

    // 设置各视口背景色
    viewer->setBackgroundColor(0.1, 0.1, 0.2, v1);  // 深蓝
    viewer->setBackgroundColor(0.1, 0.15, 0.1, v2); // 深绿

    // 颜色处理器
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        cloud_red(cloud, 255, 0, 0); // 原始点云 - 红色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        cloud_green(boundary_points, 0, 255, 0); // 边界点云 - 绿色

    // 视口1: 原始点云
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_red, "original_cloud", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "original_cloud", v1);

    viewer->addText("Original Point Cloud", 10, 20, 16, 1, 1, 1, "original_text", v1);
    viewer->addText("Red: All Points", 10, 40, 14, 1, 0, 0, "cloud_red_text", v1);

    std::string cloud_info = "Total Points: " + std::to_string(cloud->size());
    std::string file_info = "File: bunny.pcd";
    viewer->addText(cloud_info, 10, 60, 12, 1, 1, 1, "cloud_info", v1);
    viewer->addText(file_info, 10, 75, 12, 1, 1, 1, "file_info", v1);

    // 添加坐标系
    viewer->addCoordinateSystem(0.1, "axis_v1", v1);

    // 视口2: 边界点云
    viewer->addPointCloud<pcl::PointXYZ>(boundary_points, cloud_green, "boundary_cloud", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "boundary_cloud", v2);

    viewer->addText("Boundary Points", 10, 20, 16, 1, 1, 1, "boundary_text", v2);
    viewer->addText("Green: Boundary Points", 10, 40, 14, 0, 1, 0, "boundary_green_text", v2);

    std::string boundary_info = "Boundary Points: " + std::to_string(boundary_points->size());
    std::string boundary_ratio = "Boundary Ratio: " +
                                 std::to_string(static_cast<double>(boundary_points->size()) / cloud->size() * 100).substr(0, 5) + "%";
    viewer->addText(boundary_info, 10, 60, 12, 1, 1, 1, "boundary_info", v2);
    viewer->addText(boundary_ratio, 10, 75, 12, 1, 1, 1, "boundary_ratio", v2);

    // 添加坐标系
    viewer->addCoordinateSystem(0.1, "axis_v2", v2);

    // 设置相机参数
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 3, 0, 0, 0, 0, 1, 0); // 适应兔子点云

    // 主循环
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main(int argc, char **argv)
{
    std::cout << "=== Point Cloud Boundary Extraction Analysis ===" << std::endl;
    auto total_start_time = std::chrono::steady_clock::now();

    // 加载点云
    auto load_start_time = std::chrono::steady_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile("cloud.pcd", *cloud) == -1)
    {
        std::cerr << "Error: Couldn't read PCD file: cloud.pcd" << std::endl;
        return -1;
    }
    auto load_end_time = std::chrono::steady_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end_time - load_start_time);

    std::cout << "\n=== Step 1: Point Cloud Information ===" << std::endl;
    std::cout << "Loaded " << cloud->size() << " points." << std::endl;
    std::cout << "Loading time: " << load_duration.count() << " ms" << std::endl;

    // 计算法向量
    std::cout << "\n=== Step 2: Computing Normals ===" << std::endl;
    auto normal_start_time = std::chrono::steady_clock::now();

    // 使用OMP加速的法线估计器
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setNumberOfThreads(8);
    std::cout << "Using OMP with 8 threads for normal computation." << std::endl;

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    normal_estimator.setInputCloud(cloud);
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setKSearch(50); // 保持原参数

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    normal_estimator.compute(*normals);

    auto normal_end_time = std::chrono::steady_clock::now();
    auto normal_duration = std::chrono::duration_cast<std::chrono::milliseconds>(normal_end_time - normal_start_time);
    std::cout << "Normals computed: " << normals->size() << std::endl;
    std::cout << "Normal computation time: " << normal_duration.count() << " ms" << std::endl;

    // 边界特征提取
    std::cout << "\n=== Step 3: Extracting Boundary Features ===" << std::endl;
    auto boundary_start_time = std::chrono::steady_clock::now();

    pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> boundary_estimator;
    boundary_estimator.setInputCloud(cloud);
    boundary_estimator.setInputNormals(normals);
    boundary_estimator.setSearchMethod(tree);
    boundary_estimator.setRadiusSearch(1);
    boundary_estimator.setAngleThreshold(M_PI / 15);

    pcl::PointCloud<pcl::Boundary> boundaries;
    boundary_estimator.compute(boundaries);

    // 提取边界点
    pcl::PointCloud<pcl::PointXYZ>::Ptr boundary_points(new pcl::PointCloud<pcl::PointXYZ>);
    int boundary_count = 0;
    for (size_t i = 0; i < cloud->size(); ++i)
    {
        if (boundaries[i].boundary_point)
        {
            boundary_points->push_back(cloud->points[i]);
            boundary_count++;
        }
    }

    // 设置点云的宽度和高度
    boundary_points->width = boundary_points->size();
    boundary_points->height = 1;

    auto boundary_end_time = std::chrono::steady_clock::now();
    auto boundary_duration = std::chrono::duration_cast<std::chrono::milliseconds>(boundary_end_time - boundary_start_time);

    std::cout << "Boundary points extracted: " << boundary_count << std::endl;
    std::cout << "Boundary extraction time: " << boundary_duration.count() << " ms" << std::endl;
    std::cout << "Boundary ratio: " << static_cast<double>(boundary_count) / cloud->size() * 100 << "%" << std::endl;

    // 参数信息
    std::cout << "\n=== Step 4: Parameter Information ===" << std::endl;
    std::cout << "Normal estimation: KSearch = 50" << std::endl;
    std::cout << "Boundary extraction:" << std::endl;
    std::cout << "  - Radius search: 0.01" << std::endl;
    std::cout << "  - Angle threshold: " << M_PI / 15 << " radians (" << (M_PI / 15) * 180 / M_PI << " degrees)" << std::endl;

    // 总结
    auto total_end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);

    std::cout << "\n=== Analysis Summary ===" << std::endl;
    std::cout << "Loading time: " << load_duration.count() << " ms" << std::endl;
    std::cout << "Normal computation time: " << normal_duration.count() << " ms" << std::endl;
    std::cout << "Boundary extraction time: " << boundary_duration.count() << " ms" << std::endl;
    std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "\nResults:" << std::endl;
    std::cout << "  - Original points: " << cloud->size() << std::endl;
    std::cout << "  - Boundary points: " << boundary_points->size() << std::endl;
    std::cout << "  - Boundary ratio: " << static_cast<double>(boundary_points->size()) / cloud->size() * 100 << "%" << std::endl;

    // 调用可视化函数
    visualize_boundary_extraction(cloud, boundary_points, "Point Cloud Boundary Extraction - Bunny Dataset");

    return 0;
}