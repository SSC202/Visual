#include <iostream>
#include <thread>
#include <chrono>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h> // 计算主曲率
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/time.h>

using namespace std;

/**
 * @brief Visualize point cloud with normals and curvatures
 * 可视化点云及其法线和曲率
 */
void visualize_curvatures(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                          pcl::PointCloud<pcl::Normal>::Ptr &normals,
                          pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr &curvatures,
                          const std::string &window_title = "Principal Curvatures Analysis")
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0.05, 0.05, 0.15); // 深蓝色背景

    // 创建三个视口
    int v1(0), v2(1), v3(2);
    viewer->createViewPort(0.0, 0.0, 0.33, 1.0, v1);  // 原始点云
    viewer->createViewPort(0.33, 0.0, 0.66, 1.0, v2); // 法线可视化
    viewer->createViewPort(0.66, 0.0, 1.0, 1.0, v3);  // 曲率可视化

    // 设置各视口背景色
    viewer->setBackgroundColor(0.1, 0.1, 0.2, v1);
    viewer->setBackgroundColor(0.1, 0.15, 0.1, v2);
    viewer->setBackgroundColor(0.15, 0.1, 0.1, v3);

    // 颜色处理器
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        cloud_green(cloud, 0, 225, 0); // 点云 - 绿色（保持原代码颜色）
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        cloud_blue(cloud, 0, 0, 255); // 点云 - 蓝色（用于法线视图）
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        cloud_white(cloud, 255, 255, 255); // 点云 - 白色（用于曲率视图）

    // 视口1: 原始点云
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_green, "original_cloud", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original_cloud", v1);
    viewer->addText("Original Point Cloud", 10, 20, 16, 1, 1, 1, "original_text", v1);
    viewer->addText("Green: Point Cloud", 10, 40, 14, 0, 1, 0, "cloud_text", v1);

    std::string cloud_info = "Points: " + std::to_string(cloud->size());
    viewer->addText(cloud_info, 10, 60, 12, 1, 1, 1, "cloud_info", v1);

    // 视口2: 法线可视化
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_blue, "cloud_normals", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_normals", v2);
    viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 20, 2, "normals", v2);
    viewer->addText("Point Cloud Normals", 10, 20, 16, 1, 1, 1, "normals_text", v2);
    viewer->addText("Blue: Point Cloud", 10, 40, 14, 0, 0, 1, "cloud_normals_text", v2);
    viewer->addText("White: Normal Vectors", 10, 60, 14, 1, 1, 1, "normal_vectors_text", v2);
    viewer->addText("Normal Display: 1 in 20 points", 10, 80, 12, 1, 1, 1, "normal_display_info", v2);

    // 视口3: 曲率可视化
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_white, "cloud_curvatures", v3);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_curvatures", v3);

    // 尝试添加主曲率可视化
    try
    {
        viewer->addPointCloudPrincipalCurvatures<pcl::PointXYZ, pcl::Normal>(cloud, normals, curvatures, 10, 10, "curvatures", v3);
        viewer->addText("Principal Curvatures", 10, 20, 16, 1, 1, 1, "curvatures_text", v3);
        viewer->addText("White: Point Cloud", 10, 40, 14, 1, 1, 1, "cloud_curvatures_text", v3);
        viewer->addText("Red/Blue: Max/Min Curvature", 10, 60, 14, 1, 0.5, 0, "curvature_directions_text", v3);
        viewer->addText("Curvature Display: 1 in 10 points", 10, 80, 12, 1, 1, 1, "curvature_display_info", v3);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Warning: Could not visualize principal curvatures: " << e.what() << std::endl;
        viewer->addText("Principal Curvatures", 10, 20, 16, 1, 1, 1, "curvatures_text", v3);
        viewer->addText("Visualization not available", 10, 40, 14, 1, 0, 0, "error_text", v3);
    }

    // 添加坐标系
    viewer->addCoordinateSystem(0.1, "axis_v1", v1);
    viewer->addCoordinateSystem(0.1, "axis_v2", v2);
    viewer->addCoordinateSystem(0.1, "axis_v3", v3);

    // 设置相机参数
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 1, 0, 0, 0, 0, 1, 0);

    // 主循环
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main(int argc, char *argv[])
{
    std::cout << "=== Principal Curvatures Analysis ===" << std::endl;

    // 总计时开始
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

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setInputCloud(cloud);

    // 创建 KD 树用于搜索
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    normal_estimator.setSearchMethod(tree);

    // 设置搜索参数
    normal_estimator.setKSearch(10); // K 近邻搜索

    // 计算法线
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    normal_estimator.compute(*normals);

    auto normal_end_time = std::chrono::steady_clock::now();
    auto normal_duration = std::chrono::duration_cast<std::chrono::milliseconds>(normal_end_time - normal_start_time);

    std::cout << "Normal computation parameters:" << std::endl;
    std::cout << "  Search method: KdTree" << std::endl;
    std::cout << "  K nearest neighbors: 10" << std::endl;
    std::cout << "Normals computed: " << normals->size() << " normals" << std::endl;
    std::cout << "Normal computation time: " << normal_duration.count() << " ms" << std::endl;

    // 主曲率计算
    std::cout << "\n=== Step 2: Computing Principal Curvatures ===" << std::endl;

    auto curvature_start_time = std::chrono::steady_clock::now();

    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> curvature_estimator;
    curvature_estimator.setInputCloud(cloud);     // 提供原始点云
    curvature_estimator.setInputNormals(normals); // 为点云提供法线
    curvature_estimator.setSearchMethod(tree);    // 使用相同的 KD Tree

    // 设置搜索参数
    curvature_estimator.setKSearch(10);

    // 计算主曲率
    pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr curvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>());
    curvature_estimator.compute(*curvatures);

    auto curvature_end_time = std::chrono::steady_clock::now();
    auto curvature_duration = std::chrono::duration_cast<std::chrono::milliseconds>(curvature_end_time - curvature_start_time);

    std::cout << "Curvature computation parameters:" << std::endl;
    std::cout << "  Search method: KdTree" << std::endl;
    std::cout << "  K nearest neighbors: 10" << std::endl;
    std::cout << "Curvatures computed: " << curvatures->points.size() << " curvatures" << std::endl;
    std::cout << "Curvature computation time: " << curvature_duration.count() << " ms" << std::endl;

    // 显示样本点的曲率信息
    std::cout << "\n=== Sample Curvature Information ===" << std::endl;

    // 检查点云是否为空
    if (curvatures->empty())
    {
        std::cerr << "Error: No curvatures computed!" << std::endl;
        return -1;
    }

    // 显示前5个点的曲率信息
    int sample_count = std::min(5, static_cast<int>(curvatures->size()));
    for (int i = 0; i < sample_count; ++i)
    {
        std::cout << "\nPoint " << i << ":" << std::endl;
        std::cout << "  Position: (" << cloud->points[i].x << ", "
                  << cloud->points[i].y << ", "
                  << cloud->points[i].z << ")" << std::endl;
        std::cout << "  Maximum principal curvature: " << curvatures->points[i].pc1 << std::endl;
        std::cout << "  Minimum principal curvature: " << curvatures->points[i].pc2 << std::endl;
        std::cout << "  Principal curvature direction (max):" << std::endl;
        std::cout << "    X: " << curvatures->points[i].principal_curvature_x << std::endl;
        std::cout << "    Y: " << curvatures->points[i].principal_curvature_y << std::endl;
        std::cout << "    Z: " << curvatures->points[i].principal_curvature_z << std::endl;

        // 计算曲率差和高斯曲率
        double mean_curvature = (curvatures->points[i].pc1 + curvatures->points[i].pc2) / 2.0;
        double gaussian_curvature = curvatures->points[i].pc1 * curvatures->points[i].pc2;
        std::cout << "  Mean curvature: " << mean_curvature << std::endl;
        std::cout << "  Gaussian curvature: " << gaussian_curvature << std::endl;
    }

    // 计算曲率统计信息
    if (!curvatures->empty())
    {
        double max_pc1 = curvatures->points[0].pc1;
        double min_pc1 = curvatures->points[0].pc1;
        double max_pc2 = curvatures->points[0].pc2;
        double min_pc2 = curvatures->points[0].pc2;

        for (size_t i = 1; i < curvatures->size(); ++i)
        {
            if (curvatures->points[i].pc1 > max_pc1)
                max_pc1 = curvatures->points[i].pc1;
            if (curvatures->points[i].pc1 < min_pc1)
                min_pc1 = curvatures->points[i].pc1;
            if (curvatures->points[i].pc2 > max_pc2)
                max_pc2 = curvatures->points[i].pc2;
            if (curvatures->points[i].pc2 < min_pc2)
                min_pc2 = curvatures->points[i].pc2;
        }

        std::cout << "\n=== Curvature Statistics ===" << std::endl;
        std::cout << "Maximum principal curvature range: [" << min_pc1 << ", " << max_pc1 << "]" << std::endl;
        std::cout << "Minimum principal curvature range: [" << min_pc2 << ", " << max_pc2 << "]" << std::endl;
    }

    // 计算总时间
    auto total_end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);

    // 结果总结
    std::cout << "\n=== Analysis Summary ===" << std::endl;
    std::cout << "Total analysis time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "  Loading time: " << load_duration.count() << " ms" << std::endl;
    std::cout << "  Normal computation time: " << normal_duration.count() << " ms" << std::endl;
    std::cout << "  Curvature computation time: " << curvature_duration.count() << " ms" << std::endl;
    std::cout << "Points processed: " << cloud->size() << std::endl;
    std::cout << "Normals computed: " << normals->size() << std::endl;
    std::cout << "Curvatures computed: " << curvatures->size() << std::endl;

    // 可视化说明
    std::cout << "\n=== Visualization Information ===" << std::endl;
    std::cout << "1st View: Original point cloud (Green)" << std::endl;
    std::cout << "2nd View: Point cloud normals (Blue points with white normal vectors)" << std::endl;
    std::cout << "3rd View: Principal curvatures (White points with curvature direction vectors)" << std::endl;
    std::cout << "Press 'q' to exit the visualization window" << std::endl;
    std::cout << "Use mouse to rotate and scroll to zoom" << std::endl;

    // 结果可视化
    visualize_curvatures(cloud, normals, curvatures, "Principal Curvatures Analysis");

    return 0;
}