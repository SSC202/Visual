#include <iostream>
#include <thread>
#include <chrono>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h> // 使用OMP并行计算
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/time.h>

using namespace std;

/**
 * @brief Visualize point cloud with normals using two viewports
 * 使用双视口可视化点云及其法向量
 */
void visualize_normals(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                       pcl::PointCloud<pcl::Normal>::Ptr &normals,
                       const std::string &window_title = "Point Cloud Normals Analysis")
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0.05, 0.05, 0.15); // 深蓝色背景

    // 创建两个视口
    int v1(0), v2(1);
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1); // 原始点云
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2); // 法向量可视化

    // 设置各视口背景色
    viewer->setBackgroundColor(0.1, 0.1, 0.2, v1);
    viewer->setBackgroundColor(0.1, 0.15, 0.1, v2);

    // 颜色处理器
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        cloud_green(cloud, 0, 225, 0); // 点云 - 绿色（保持原代码颜色）
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        cloud_white(cloud, 255, 255, 255); // 点云 - 白色（用于法线视图）
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        cloud_red(cloud, 255, 0, 0); // 点云 - 红色（备选视图）

    // 视口1: 原始点云
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_green, "original_cloud", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud", v1);
    viewer->addText("Original Point Cloud", 10, 20, 16, 1, 1, 1, "original_text", v1);
    viewer->addText("Green: Point Cloud", 10, 40, 14, 0, 1, 0, "cloud_text", v1);

    std::string cloud_info = "Points: " + std::to_string(cloud->size());
    std::string file_info = "File: 车载点云.pcd";
    viewer->addText(cloud_info, 10, 60, 12, 1, 1, 1, "cloud_info", v1);
    viewer->addText(file_info, 10, 75, 12, 1, 1, 1, "file_info", v1);

    // 视口2: 法向量可视化
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_green, "cloud_normals", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_normals", v2);
    viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 10, 0.2, "normals", v2);
    viewer->addText("Point Cloud with Normals", 10, 20, 16, 1, 1, 1, "normals_text", v2);
    viewer->addText("Green: Point Cloud", 10, 40, 14, 1, 1, 1, "cloud_normals_text", v2);
    viewer->addText("White: Normal Vectors", 10, 60, 14, 0, 1, 0, "normal_vectors_text", v2);

    std::string normal_info = "Normal Display: 1 in 10 points";
    std::string normal_length = "Normal Length: 0.52";
    viewer->addText(normal_info, 10, 80, 12, 1, 1, 1, "normal_display_info", v2);
    viewer->addText(normal_length, 10, 95, 12, 1, 1, 1, "normal_length_info", v2);

    // 添加坐标系
    viewer->addCoordinateSystem(0.1, "axis_v1", v1);
    viewer->addCoordinateSystem(0.1, "axis_v2", v2);

    // 设置相机参数
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 50, 0, 0, 0, 0, 1, 0); // 调整相机位置以适应车载点云

    // 主循环
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main()
{
    std::cout << "=== Point Cloud Normal Estimation (OMP Accelerated) ===" << std::endl;
    auto total_start_time = std::chrono::steady_clock::now();

    // 加载点云数据
    auto load_start_time = std::chrono::steady_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("cloud.pcd", *cloud) == -1)
    {
        std::cerr << "Error: Couldn't read PCD file: cloud.pcd" << std::endl;
        return -1;
    }
    auto load_end_time = std::chrono::steady_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end_time - load_start_time);
    std::cout << "\n=== Point Cloud Information ===" << std::endl;
    std::cout << "Loaded " << cloud->size() << " points." << std::endl;
    std::cout << "Loading time: " << load_duration.count() << " ms" << std::endl;

    // 计算点云的法线
    std::cout << "\n=== Step 1: Computing Point Cloud Normals ===" << std::endl;
    auto normal_start_time = std::chrono::steady_clock::now();

    // 创建 OMP 加速的法线估计器
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator;

    int threads_count = 10; // 定义一个变量来记录我们使用的线程数
    normal_estimator.setNumberOfThreads(threads_count);
    std::cout << "Using OMP with " << threads_count << " threads." << std::endl; // 使用变量

    normal_estimator.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setKSearch(10); // 保持原参数

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    normal_estimator.compute(*normals); // 计算法线

    auto normal_end_time = std::chrono::steady_clock::now();
    auto normal_duration = std::chrono::duration_cast<std::chrono::milliseconds>(normal_end_time - normal_start_time);
    std::cout << "Normals computed: " << normals->size() << std::endl;
    std::cout << "Normal computation time: " << normal_duration.count() << " ms" << std::endl;

    // 检查并清理法线中的 NaN 值
    int nan_count = 0;
    for (auto &normal : normals->points)
    {
        // 如果法线的模长是 NaN, 或者法线三个分量中有任意一个是NaN/无穷大
        if (!pcl::isFinite(normal))
        {
            nan_count++;
            // 将无效的法线设置为一个默认值, 以便可视化
            normal.normal_x = normal.normal_y = normal.normal_z = 0.0;
            normal.curvature = 0.0;
        }
    }
    if (nan_count > 0)
    {
        std::cout << "\nWarning: Found and fixed " << nan_count
                  << " invalid normals (containing NaN/Inf). They are set to zero vector for display."
                  << std::endl;
    }
    else
    {
        std::cout << "All normals are valid." << std::endl;
    }

    // 计算总时间并输出
    auto total_end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);
    std::cout << "\n=== Analysis Summary ===" << std::endl;
    std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;

    // 调用可视化函数
    visualize_normals(cloud, normals, "Point Cloud Normals Analysis (NaN Fixed)");

    return 0;
}