#include <iostream>
#include <thread>
#include <chrono>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>               // NDT配准
#include <pcl/filters/approximate_voxel_grid.h> // 体素滤波
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/time.h>

using namespace std;

/**
 * @brief Visualize registration results with four viewports
 * 使用四视口可视化配准结果
 */
void visualize_registration(pcl::PointCloud<pcl::PointXYZ>::Ptr &source,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr &target,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_source,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_target,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr &ndt_result,
                            const std::string &window_title = "NDT Registration Results")
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0.05, 0.05, 0.15); // 深蓝色背景

    // 创建四个视口
    int v1(0), v2(1), v3(2), v4(3);
    viewer->createViewPort(0.0, 0.0, 0.25, 1.0, v1); // 原始点云
    viewer->createViewPort(0.25, 0.0, 0.5, 1.0, v2); // 滤波后的点云
    viewer->createViewPort(0.5, 0.0, 0.75, 1.0, v3); // NDT配准结果
    viewer->createViewPort(0.75, 0.0, 1.0, 1.0, v4); // 最终重叠效果

    // 设置各视口背景色
    viewer->setBackgroundColor(0.1, 0.1, 0.2, v1);
    viewer->setBackgroundColor(0.1, 0.15, 0.1, v2);
    viewer->setBackgroundColor(0.15, 0.1, 0.1, v3);
    viewer->setBackgroundColor(0.1, 0.1, 0.1, v4);

    // 颜色处理器
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        source_red(source, 255, 0, 0); // 源点云 - 红色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        target_blue(target, 0, 0, 255); // 目标点云 - 蓝色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        filtered_source_cyan(filtered_source, 0, 255, 255); // 滤波后源点云 - 青色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        filtered_target_magenta(filtered_target, 255, 0, 255); // 滤波后目标点云 - 洋红色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        ndt_green(ndt_result, 0, 255, 0); // NDT结果 - 绿色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        target_white(target, 255, 255, 255); // 目标点云 - 白色（用于重叠视图）

    // 视口1: 原始点云
    viewer->addPointCloud<pcl::PointXYZ>(source, source_red, "source_cloud", v1);
    viewer->addPointCloud<pcl::PointXYZ>(target, target_blue, "target_cloud", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source_cloud", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target_cloud", v1);
    viewer->addText("Original Point Clouds", 10, 20, 16, 1, 1, 1, "original_text", v1);
    viewer->addText("Red: Source Cloud", 10, 40, 14, 1, 0, 0, "source_text", v1);
    viewer->addText("Blue: Target Cloud", 10, 60, 14, 0, 0, 1, "target_text", v1);

    std::string source_info = "Source: " + std::to_string(source->size()) + " points";
    std::string target_info = "Target: " + std::to_string(target->size()) + " points";
    viewer->addText(source_info, 10, 80, 12, 1, 1, 1, "source_info", v1);
    viewer->addText(target_info, 10, 95, 12, 1, 1, 1, "target_info", v1);

    // 视口2: 滤波后的点云
    viewer->addPointCloud<pcl::PointXYZ>(filtered_source, filtered_source_cyan, "filtered_source_cloud", v2);
    viewer->addPointCloud<pcl::PointXYZ>(filtered_target, filtered_target_magenta, "filtered_target_cloud", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "filtered_source_cloud", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "filtered_target_cloud", v2);
    viewer->addText("Filtered Point Clouds", 10, 20, 16, 1, 1, 1, "filtered_text", v2);
    viewer->addText("Cyan: Filtered Source", 10, 40, 14, 0, 1, 1, "filtered_source_text", v2);
    viewer->addText("Magenta: Filtered Target", 10, 60, 14, 1, 0, 1, "filtered_target_text", v2);

    std::string filtered_source_info = "Filtered Source: " + std::to_string(filtered_source->size()) + " points";
    std::string filtered_target_info = "Filtered Target: " + std::to_string(filtered_target->size()) + " points";
    viewer->addText(filtered_source_info, 10, 80, 12, 1, 1, 1, "filtered_source_info", v2);
    viewer->addText(filtered_target_info, 10, 95, 12, 1, 1, 1, "filtered_target_info", v2);
    viewer->addText("Voxel Size: 0.2", 10, 110, 12, 1, 1, 1, "voxel_info", v2);

    // 视口3: NDT配准结果
    viewer->addPointCloud<pcl::PointXYZ>(target, target_blue, "target_cloud_v3", v3);
    viewer->addPointCloud<pcl::PointXYZ>(ndt_result, ndt_green, "ndt_result_cloud", v3);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target_cloud_v3", v3);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "ndt_result_cloud", v3);
    viewer->addText("NDT Registration Result", 10, 20, 16, 1, 1, 1, "ndt_text", v3);
    viewer->addText("Blue: Target", 10, 40, 14, 0, 0, 1, "target_text_v3", v3);
    viewer->addText("Green: NDT Result", 10, 60, 14, 0, 1, 0, "ndt_result_text", v3);

    std::string ndt_info = "NDT Resolution: 1.0";
    std::string step_info = "Step Size: 0.1";
    viewer->addText(ndt_info, 10, 80, 12, 1, 1, 1, "ndt_info", v3);
    viewer->addText(step_info, 10, 95, 12, 1, 1, 1, "step_info", v3);

    // 视口4: 最终重叠效果
    viewer->addPointCloud<pcl::PointXYZ>(target, target_white, "target_cloud_v4", v4);
    viewer->addPointCloud<pcl::PointXYZ>(ndt_result, ndt_green, "final_cloud_v4", v4);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 0.5, "target_cloud_v4", v4);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "final_cloud_v4", v4);
    viewer->addText("Final Overlap View", 10, 20, 16, 1, 1, 1, "overlap_text", v4);
    viewer->addText("White: Target", 10, 40, 14, 1, 1, 1, "target_text_v4", v4);
    viewer->addText("Green: NDT Result", 10, 60, 14, 0, 1, 0, "final_text_v4", v4);

    // 添加坐标系
    viewer->addCoordinateSystem(0.5, "axis_v1", v1);
    viewer->addCoordinateSystem(0.5, "axis_v2", v2);
    viewer->addCoordinateSystem(0.5, "axis_v3", v3);
    viewer->addCoordinateSystem(0.5, "axis_v4", v4);

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

int main(int argc, char *argv[])
{
    std::cout << "=== NDT (Normal Distributions Transform) Registration ===" << std::endl;
    std::cout << "PCL Version: 1.15.0" << std::endl;

    // 总计时开始
    auto total_start_time = std::chrono::steady_clock::now();

    // 加载源点云
    auto load_start_time = std::chrono::steady_clock::now();

    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("source_cloud.pcd", *source_cloud) == -1)
    {
        std::cerr << "Error: Couldn't read source PCD file: source_cloud.pcd" << std::endl;
        return -1;
    }

    // 加载目标点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("target_cloud.pcd", *target_cloud) == -1)
    {
        std::cerr << "Error: Couldn't read target PCD file: target_cloud.pcd" << std::endl;
        return -1;
    }

    auto load_end_time = std::chrono::steady_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end_time - load_start_time);

    if (source_cloud->empty() || target_cloud->empty())
    {
        cout << "Error: Point cloud files are empty!" << endl;
        return -1;
    }

    std::cout << "\n=== Point Cloud Information ===" << std::endl;
    std::cout << "Source point cloud: " << source_cloud->size() << " points" << std::endl;
    std::cout << "Target point cloud: " << target_cloud->size() << " points" << std::endl;
    std::cout << "Loading time: " << load_duration.count() << " ms" << std::endl;

    // 体素滤波预处理
    std::cout << "\n=== Step 1: Point Cloud Downsampling ===" << std::endl;

    auto filter_start_time = std::chrono::steady_clock::now();

    // 创建滤波后的点云指针
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_source(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_target(new pcl::PointCloud<pcl::PointXYZ>);

    // 配置体素滤波器
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(0.2f, 0.2f, 0.2f); // 设置体素大小

    // 滤波源点云
    voxel_filter.setInputCloud(source_cloud);
    voxel_filter.filter(*filtered_source);

    // 滤波目标点云
    voxel_filter.setInputCloud(target_cloud);
    voxel_filter.filter(*filtered_target);

    auto filter_end_time = std::chrono::steady_clock::now();
    auto filter_duration = std::chrono::duration_cast<std::chrono::milliseconds>(filter_end_time - filter_start_time);

    std::cout << "Filtering Parameters:" << std::endl;
    std::cout << "  Voxel Size: 0.2 x 0.2 x 0.2" << std::endl;
    std::cout << "Filtered source cloud: " << filtered_source->size() << " points" << std::endl;
    std::cout << "Filtered target cloud: " << filtered_target->size() << " points" << std::endl;
    std::cout << "Filtering time: " << filter_duration.count() << " ms" << std::endl;

    // 计算降采样率
    float source_downsample_rate = 100.0f * (1.0f - static_cast<float>(filtered_source->size()) / source_cloud->size());
    float target_downsample_rate = 100.0f * (1.0f - static_cast<float>(filtered_target->size()) / target_cloud->size());
    std::cout << "Downsampling rate - Source: " << source_downsample_rate << "%" << std::endl;
    std::cout << "Downsampling rate - Target: " << target_downsample_rate << "%" << std::endl;

    // NDT配准
    std::cout << "\n=== Step 2: Performing NDT Registration ===" << std::endl;

    auto ndt_start_time = std::chrono::steady_clock::now();

    // 创建NDT配准对象
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;

    // 设置NDT参数
    ndt.setStepSize(0.1);               // 为More-Thuente线搜索设置最大步长
    ndt.setResolution(1.0f);            // 设置NDT网格结构的分辨率
    ndt.setMaximumIterations(50);       // 设置最大迭代次数
    ndt.setTransformationEpsilon(0.01); // 设置终止条件的最小转换差异

    // 设置输入点云
    ndt.setInputSource(filtered_source);
    ndt.setInputTarget(filtered_target);

    std::cout << "NDT Parameters:" << std::endl;
    std::cout << "  Step Size: 0.1" << std::endl;
    std::cout << "  Resolution: 1.0" << std::endl;
    std::cout << "  Maximum Iterations: 50" << std::endl;
    std::cout << "  Transformation Epsilon: 0.01" << std::endl;

    // 执行NDT配准
    pcl::PointCloud<pcl::PointXYZ>::Ptr ndt_result_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    ndt.align(*ndt_result_filtered);

    auto ndt_end_time = std::chrono::steady_clock::now();
    auto ndt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(ndt_end_time - ndt_start_time);

    // 检查NDT配准是否成功
    if (!ndt.hasConverged())
    {
        std::cerr << "\nNDT registration failed to converge!" << std::endl;
        return -1;
    }

    std::cout << "NDT registration completed in: " << ndt_duration.count() << " ms" << std::endl;
    std::cout << "NDT has converged: " << (ndt.hasConverged() ? "Yes" : "No") << std::endl;
    std::cout << "NDT fitness score: " << ndt.getFitnessScore() << std::endl;
    std::cout << "Applied " << ndt.getFinalNumIteration() << " iterations" << std::endl;

    // 获取NDT变换矩阵
    Eigen::Matrix4f ndt_transformation = ndt.getFinalTransformation();
    std::cout << "NDT transformation matrix:" << std::endl;
    std::cout << ndt_transformation << std::endl;

    // 将变换应用到原始源点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr ndt_result_original(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*source_cloud, *ndt_result_original, ndt_transformation);

    // 保存转换后的点云
    std::cout << "\n=== Step 3: Saving Results ===" << std::endl;
    if (pcl::io::savePCDFileBinary("aligned_cloud.pcd", *ndt_result_original) == 0)
    {
        std::cout << "Registered point cloud saved to: aligned_cloud.pcd" << std::endl;
    }
    else
    {
        std::cerr << "Failed to save registered point cloud!" << std::endl;
    }

    // 计算总时间
    auto total_end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);

    // 结果总结
    std::cout << "\n=== Registration Summary ===" << std::endl;
    std::cout << "Total registration time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "  Loading time: " << load_duration.count() << " ms" << std::endl;
    std::cout << "  Filtering time: " << filter_duration.count() << " ms" << std::endl;
    std::cout << "  NDT time: " << ndt_duration.count() << " ms" << std::endl;
    std::cout << "Final fitness score: " << ndt.getFitnessScore() << std::endl;

    // 质量评估
    float fitness_score = ndt.getFitnessScore();
    if (fitness_score < 0.001f)
    {
        std::cout << "Registration quality: Excellent" << std::endl;
    }
    else if (fitness_score < 0.005f)
    {
        std::cout << "Registration quality: Very Good" << std::endl;
    }
    else if (fitness_score < 0.01f)
    {
        std::cout << "Registration quality: Good" << std::endl;
    }
    else if (fitness_score < 0.05f)
    {
        std::cout << "Registration quality: Acceptable" << std::endl;
    }
    else
    {
        std::cout << "Registration quality: Poor - consider parameter adjustment" << std::endl;
    }

    // 可视化说明
    std::cout << "\n=== Visualization Information ===" << std::endl;
    std::cout << "1st View: Original point clouds (Red: Source, Blue: Target)" << std::endl;
    std::cout << "2nd View: Filtered point clouds (Cyan: Filtered Source, Magenta: Filtered Target)" << std::endl;
    std::cout << "3rd View: NDT registration result (Blue: Target, Green: NDT Result)" << std::endl;
    std::cout << "4th View: Final overlap (White: Target, Green: NDT Result)" << std::endl;
    std::cout << "Press 'q' to exit the visualization window" << std::endl;
    std::cout << "Use mouse to rotate and scroll to zoom" << std::endl;

    // 结果可视化
    visualize_registration(source_cloud, target_cloud, filtered_source,
                           filtered_target, ndt_result_original,
                           "NDT Registration Results");

    return 0;
}