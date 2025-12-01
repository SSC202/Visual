#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/ia_fpcs.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <thread>
#include <chrono>

int main()
{
    // 定义点云类型和对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 读取源点云和目标点云
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("source_cloud.pcd", *source_cloud) == -1)
    {
        PCL_ERROR("Could not read source point cloud file!\n");
        return -1;
    }
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("target_cloud.pcd", *target_cloud) == -1)
    {
        PCL_ERROR("Could not read target point cloud file!\n");
        return -1;
    }

    std::cout << "=== 4PCS Point Cloud Registration ===" << std::endl;
    std::cout << "Source point cloud: " << source_cloud->size() << " points" << std::endl;
    std::cout << "Target point cloud: " << target_cloud->size() << " points" << std::endl;

    // 创建 4PCS 配准对象
    pcl::registration::FPCSInitialAlignment<pcl::PointXYZ, pcl::PointXYZ> fpcs;

    // 设置输入源点云和目标点云
    fpcs.setInputSource(source_cloud);
    fpcs.setInputTarget(target_cloud);

    // 设置4PCS算法参数
    fpcs.setApproxOverlap(0.55);     // 设置源点云和目标点云之间大致的重叠度
    fpcs.setDelta(0.05);            // 设置配准后源点云和目标点云之间的距离
    fpcs.setNumberOfSamples(1000);  // 设置验证配准效果时使用的采样点数量
    fpcs.setMaxComputationTime(50); // 设置最大计算时间

    std::cout << "\n=== Performing 4PCS Registration ===" << std::endl;
    std::cout << "Parameters: ApproxOverlap=0.55, Delta=0.05, NumberOfSamples=1000" << std::endl;

    // 执行配准
    fpcs.align(*aligned_cloud);

    // 输出配准结果
    if (fpcs.hasConverged())
    {
        std::cout << "\n4PCS registration successful!" << std::endl;
        std::cout << "Fitness score: " << fpcs.getFitnessScore() << std::endl;
        std::cout << "Transformation matrix:" << std::endl;
        std::cout << fpcs.getFinalTransformation() << std::endl;

        // 保存配准后的点云
        pcl::io::savePCDFile<pcl::PointXYZ>("aligned_cloud.pcd", *aligned_cloud);
        std::cout << "Aligned point cloud saved to 'aligned_cloud.pcd'" << std::endl;
    }
    else
    {
        std::cout << "\n4PCS registration failed!" << std::endl;
        return -1;
    }

    // 可视化

    // 创建可视化器
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("4PCS Point Cloud Registration"));
    viewer->setBackgroundColor(0.05, 0.05, 0.15); // 深蓝色背景

    // 左侧视口 - 配准前
    int v1(0);
    viewer->createViewPort(0.0, 0.0, 0.33, 1.0, v1);
    viewer->setBackgroundColor(0.1, 0.1, 0.2, v1);

    // 添加源点云
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_red(source_cloud, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(source_cloud, source_red, "source_cloud", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source_cloud", v1);

    // 添加目标点云
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_green(target_cloud, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(target_cloud, target_green, "target_cloud", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target_cloud", v1);

    // 添加文本说明
    viewer->addText("Before Registration", 10, 20, 16, 1, 1, 1, "before_text", v1);
    viewer->addText("Red: Source cloud", 10, 40, 14, 1, 0, 0, "source_text", v1);
    viewer->addText("Green: Target cloud", 10, 60, 14, 0, 1, 0, "target_text", v1);

    // 中间视口 - 配准后
    int v2(0);
    viewer->createViewPort(0.33, 0.0, 0.66, 1.0, v2);
    viewer->setBackgroundColor(0.1, 0.15, 0.1, v2);

    // 添加目标点云
    viewer->addPointCloud<pcl::PointXYZ>(target_cloud, target_green, "target_cloud_v2", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target_cloud_v2", v2);

    // 添加配准后的点云
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligned_blue(aligned_cloud, 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ>(aligned_cloud, aligned_blue, "aligned_cloud", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "aligned_cloud", v2);

    // 添加文本说明
    viewer->addText("After Registration", 10, 20, 16, 1, 1, 1, "after_text", v2);
    viewer->addText("Green: Target cloud", 10, 40, 14, 0, 1, 0, "target_text_v2", v2);
    viewer->addText("Blue: Aligned cloud", 10, 60, 14, 0, 0, 1, "aligned_text", v2);

    // 右侧视口 - 重叠效果
    int v3(0);
    viewer->createViewPort(0.66, 0.0, 1.0, 1.0, v3);
    viewer->setBackgroundColor(0.15, 0.1, 0.1, v3);

    // 添加目标点云
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_white(target_cloud, 255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZ>(target_cloud, target_white, "target_cloud_v3", v3);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target_cloud_v3", v3);

    // 添加配准后的点云
    viewer->addPointCloud<pcl::PointXYZ>(aligned_cloud, aligned_blue, "aligned_cloud_v3", v3);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "aligned_cloud_v3", v3);

    // 添加文本说明
    viewer->addText("Overlap View", 10, 20, 16, 1, 1, 1, "overlap_text", v3);
    viewer->addText("White: Target cloud", 10, 40, 14, 1, 1, 1, "target_text_v3", v3);
    viewer->addText("Blue: Aligned cloud", 10, 60, 14, 0, 0, 1, "aligned_text_v3", v3);
    std::string fitness_info = "Fitness: " + std::to_string(fpcs.getFitnessScore());
    viewer->addText(fitness_info, 10, 80, 14, 1, 1, 1, "fitness_text", v3);

    // 公共设置

    // 添加标题
    viewer->addText("4PCS Point Cloud Registration", 400, 20, 18, 1, 1, 1, "title");

    // 添加坐标系
    viewer->addCoordinateSystem(0.5, "axis_v1", v1);
    viewer->addCoordinateSystem(0.5, "axis_v2", v2);
    viewer->addCoordinateSystem(0.5, "axis_v3", v3);

    // 设置相机参数
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 5, 0, 0, 0, 0, 1, 0);

    std::cout << "\n=== Visualization Started ===" << std::endl;
    std::cout << "Left: Before registration (Red: Source, Green: Target)" << std::endl;
    std::cout << "Middle: After registration (Green: Target, Blue: Aligned)" << std::endl;
    std::cout << "Right: Overlap view (White: Target, Blue: Aligned)" << std::endl;
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