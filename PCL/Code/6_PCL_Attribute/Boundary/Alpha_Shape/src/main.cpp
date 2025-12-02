#include <iostream>
#include <thread>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/surface/concave_hull.h> 
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common.h>
#include <pcl/common/time.h>

using namespace std;

/**
 * @brief Visualize concave hull extraction results using dual viewports
 * 使用双视口可视化凹包提取结果
 */
void visualize_concave_hull(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr &cloudHull,
                            double alpha_value,
                            double processing_time,
                            const std::string &window_title = "Concave Hull Extraction")
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0.05, 0.05, 0.15); // 深蓝色背景

    // 创建两个视口
    int v1(0), v2(1);
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1); // 左：原始点云
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2); // 右：凹包提取结果

    // 设置各视口背景色
    viewer->setBackgroundColor(0.1, 0.1, 0.2, v1);  // 深蓝
    viewer->setBackgroundColor(0.1, 0.15, 0.1, v2); // 深绿

    // 颜色处理器
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        cloud_green(cloud, 0, 225, 0); // 原始点云 - 绿色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        hull_red(cloudHull, 255, 0, 0); // 凹包点云 - 红色

    // 视口1: 原始点云
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_green, "original_cloud", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud", v1);

    viewer->addText("Original Point Cloud", 10, 20, 16, 1, 1, 1, "original_text", v1);
    viewer->addText("Green: All Points", 10, 40, 14, 0, 1, 0, "cloud_green_text", v1);

    std::string cloud_info = "Total Points: " + std::to_string(cloud->size());
    std::string file_info = "File: cloud.pcd";
    viewer->addText(cloud_info, 10, 60, 12, 1, 1, 1, "cloud_info", v1);
    viewer->addText(file_info, 10, 75, 12, 1, 1, 1, "file_info", v1);

    // 添加坐标系
    viewer->addCoordinateSystem(0.1, "axis_v1", v1);

    // 视口2: 凹包提取结果
    // 首先显示原始点云作为背景参考
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        cloud_gray(cloud, 100, 100, 100); // 原始点云 - 灰色作为背景
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_gray, "background_cloud", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "background_cloud", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, "background_cloud", v2);

    // 添加凹包点云（红色）
    viewer->addPointCloud<pcl::PointXYZ>(cloudHull, hull_red, "hull_cloud", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "hull_cloud", v2);

    // 添加凹包多边形（白色线框）
    viewer->addPolygon<pcl::PointXYZ>(cloudHull, 1.0, 1.0, 1.0, "concave_hull_polygon", v2);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "concave_hull_polygon", v2);

    viewer->addText("Concave Hull Extraction", 10, 20, 16, 1, 1, 1, "hull_text", v2);
    viewer->addText("Red: Hull Boundary Points", 10, 40, 14, 1, 0, 0, "hull_red_text", v2);
    viewer->addText("White: Hull Polygon", 10, 55, 14, 1, 1, 1, "hull_polygon_text", v2);

    std::string hull_info = "Hull Points: " + std::to_string(cloudHull->size());
    std::string alpha_info = "Alpha Value: " + std::to_string(alpha_value);
    std::string time_info = "Processing Time: " + std::to_string(processing_time) + " ms";
    viewer->addText(hull_info, 10, 75, 12, 1, 1, 1, "hull_info", v2);
    viewer->addText(alpha_info, 10, 90, 12, 1, 1, 1, "alpha_info", v2);
    viewer->addText(time_info, 10, 105, 12, 1, 1, 1, "time_info", v2);

    std::string save_info = "Saved to: hull.pcd";
    viewer->addText(save_info, 10, 120, 12, 0, 1, 1, "save_info", v2);

    // 添加坐标系
    viewer->addCoordinateSystem(0.1, "axis_v2", v2);

    // 设置相机参数
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 10, 0, 0, 0, 0, 1, 0);

    // 主循环
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main()
{
    std::cout << "=== Point Cloud Concave Hull (Alpha-Shape) Extraction ===" << std::endl;
    auto total_start_time = std::chrono::steady_clock::now();

    // 加载点云
    auto load_start_time = std::chrono::steady_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile("cloud.pcd", *cloud) == -1)
    {
        std::cerr << "Error: Couldn't read PCD file: cloud.pcd" << std::endl;
        std::cerr << "Please ensure the file exists in the current directory." << std::endl;
        return -1;
    }

    auto load_end_time = std::chrono::steady_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end_time - load_start_time);

    std::cout << "\n=== Step 1: Point Cloud Information ===" << std::endl;
    std::cout << "Loaded " << cloud->size() << " points." << std::endl;
    std::cout << "Loading time: " << load_duration.count() << " ms" << std::endl;

    // 分析点云范围，帮助确定合适的alpha值
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);
    float cloud_size = std::max({max_pt.x - min_pt.x, max_pt.y - min_pt.y, max_pt.z - min_pt.z});
    std::cout << "Cloud bounding box: [" << min_pt.x << ", " << max_pt.x << "] x ["
              << min_pt.y << ", " << max_pt.y << "] x ["
              << min_pt.z << ", " << max_pt.z << "]" << std::endl;
    std::cout << "Cloud size (max dimension): " << cloud_size << std::endl;

    // 计算点云的平均密度，帮助确定alpha值
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    std::vector<int> indices(2);
    std::vector<float> distances(2);
    double avg_distance = 0.0;
    int valid_points = 0;

    for (size_t i = 0; i < cloud->size(); i += 100) // 抽样计算，每100个点取一个
    {
        if (tree->nearestKSearch(cloud->points[i], 2, indices, distances) > 1)
        {
            avg_distance += std::sqrt(distances[1]); // 第二个最近邻的距离
            valid_points++;
        }
    }

    if (valid_points > 0)
    {
        avg_distance /= valid_points;
        std::cout << "Average nearest neighbor distance: " << avg_distance << std::endl;
        std::cout << "Suggested alpha range: " << avg_distance * 2 << " to " << avg_distance * 10 << std::endl;
    }

    // 计算凹包
    std::cout << "\n=== Step 2: Computing Concave Hull (Alpha-Shape) ===" << std::endl;
    std::cout << "Note: Concave hull is suitable for planar or 2.5D point clouds." << std::endl;
    std::cout << "For 3D point clouds, consider using convex hull or other methods." << std::endl;

    auto hull_start_time = std::chrono::steady_clock::now();

    // 设置凹包提取参数 - 基于点云密度自动调整alpha值
    float alpha_value = avg_distance * 5; // 基于平均距离的5倍作为默认alpha值
    if (alpha_value < 0.01)
        alpha_value = 0.01; // 设置最小阈值
    if (alpha_value > 1.0)
        alpha_value = 1.0; // 设置最大阈值

    std::cout << "Using alpha value: " << alpha_value << " (auto-adjusted)" << std::endl;
    std::cout << "Tip: Smaller alpha values create more concave shapes." << std::endl;
    std::cout << "     Larger alpha values create more convex shapes." << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudHull(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ConcaveHull<pcl::PointXYZ> concave_hull;

    concave_hull.setInputCloud(cloud);      // 输入点云
    concave_hull.setAlpha(alpha_value);     // 设置alpha值
    concave_hull.setKeepInformation(false); // 是否保留原始点云信息

    try
    {
        concave_hull.reconstruct(*cloudHull);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error during concave hull reconstruction: " << e.what() << std::endl;
        std::cerr << "Try adjusting the alpha value or ensure the point cloud is suitable for concave hull." << std::endl;

        // 尝试使用固定值重新计算
        std::cout << "Trying with fixed alpha value: 0.1" << std::endl;
        concave_hull.setAlpha(0.1);
        concave_hull.reconstruct(*cloudHull);
    }

    auto hull_end_time = std::chrono::steady_clock::now();
    auto hull_duration = std::chrono::duration_cast<std::chrono::milliseconds>(hull_end_time - hull_start_time);

    std::cout << "Concave hull computed successfully." << std::endl;
    std::cout << "Hull points: " << cloudHull->size() << std::endl;
    std::cout << "Processing time: " << hull_duration.count() << " ms" << std::endl;

    if (cloudHull->size() < 3)
    {
        std::cerr << "Warning: Concave hull contains less than 3 points." << std::endl;
        std::cerr << "This may indicate an issue with the alpha parameter or point cloud density." << std::endl;
        std::cerr << "Trying with a larger alpha value..." << std::endl;

        // 尝试使用更大的alpha值
        concave_hull.setAlpha(alpha_value * 2);
        concave_hull.reconstruct(*cloudHull);
        std::cout << "New hull points with alpha=" << alpha_value * 2 << ": " << cloudHull->size() << std::endl;
    }

    // 保存结果
    std::cout << "\n=== Step 3: Saving Results ===" << std::endl;
    auto save_start_time = std::chrono::steady_clock::now();

    if (pcl::io::savePCDFile("hull.pcd", *cloudHull, false) == 0)
    {
        std::cout << "Concave hull points saved to hull.pcd successfully." << std::endl;
    }
    else
    {
        std::cerr << "Error: Failed to save concave hull points." << std::endl;
    }

    auto save_end_time = std::chrono::steady_clock::now();
    auto save_duration = std::chrono::duration_cast<std::chrono::milliseconds>(save_end_time - save_start_time);
    std::cout << "Saving time: " << save_duration.count() << " ms" << std::endl;

    // 算法解释
    std::cout << "\n=== Step 4: Algorithm Explanation ===" << std::endl;
    std::cout << "Concave Hull (Alpha-Shape) Algorithm:" << std::endl;
    std::cout << "1. Creates a concave boundary around point cloud." << std::endl;
    std::cout << "2. Alpha parameter controls the 'concaveness':" << std::endl;
    std::cout << "   - Small alpha: More concave, captures more details" << std::endl;
    std::cout << "   - Large alpha: More convex, smoother boundary" << std::endl;
    std::cout << "3. Best for planar or near-planar point clouds." << std::endl;
    std::cout << "4. For 3D point clouds, use convex hull instead." << std::endl;

    // 总结
    auto total_end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);

    std::cout << "\n=== Analysis Summary ===" << std::endl;
    std::cout << "Loading time: " << load_duration.count() << " ms" << std::endl;
    std::cout << "Concave hull computation time: " << hull_duration.count() << " ms" << std::endl;
    std::cout << "Saving time: " << save_duration.count() << " ms" << std::endl;
    std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "\nResults:" << std::endl;
    std::cout << "  - Original points: " << cloud->size() << std::endl;
    std::cout << "  - Hull points: " << cloudHull->size() << std::endl;
    std::cout << "  - Alpha value: " << alpha_value << std::endl;
    std::cout << "  - Reduction ratio: "
              << static_cast<double>(cloudHull->size()) / cloud->size() * 100 << "%" << std::endl;

    // 调用可视化函数
    visualize_concave_hull(cloud, cloudHull, alpha_value, hull_duration.count(),
                           "Concave Hull (Alpha-Shape) Extraction");

    return 0;
}