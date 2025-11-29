#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <thread>
#include <chrono>

int main()
{
    // 创建点云指针
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);           // 原始点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected(new pcl::PointCloud<pcl::PointXYZ>); // 投影点云

    // 读取点云数据
    std::cout << "=== Loading Point Cloud ===" << std::endl;
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("test.pcd", *cloud) == -1)
    {
        PCL_ERROR("Could not read point cloud file!\n");
        return -1;
    }
    std::cout << "Original point cloud: " << cloud->width * cloud->height
              << " points" << std::endl;

    // 参数化模型投影
    std::cout << "\n=== Performing Plane Projection ===" << std::endl;

    // 创建平面模型系数
    // 平面方程: ax + by + cz + d = 0 => x + y + z = 0
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    coefficients->values.resize(4);
    coefficients->values[0] = 1.0; // x 系数 (a)
    coefficients->values[1] = 1.0; // y 系数 (b)
    coefficients->values[2] = 1.0; // z 系数 (c)
    coefficients->values[3] = 0.0; // 常数项 (d)

    std::cout << "Plane coefficients: "
              << coefficients->values[0] << "x + "
              << coefficients->values[1] << "y + "
              << coefficients->values[2] << "z + "
              << coefficients->values[3] << " = 0" << std::endl;

    // 创建投影滤波器
    pcl::ProjectInliers<pcl::PointXYZ> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);  // 设置投影模型为平面
    proj.setInputCloud(cloud);               // 设置输入点云
    proj.setModelCoefficients(coefficients); // 设置模型系数
    proj.filter(*cloud_projected);           // 执行投影滤波

    std::cout << "Projected point cloud: " << cloud_projected->width * cloud_projected->height
              << " points" << std::endl;

    // 保存投影点云
    std::cout << "\n=== Saving Projected Point Cloud ===" << std::endl;

    if (cloud_projected->empty())
    {
        PCL_ERROR("Projected point cloud is empty!\n");
        return -1;
    }
    else
    {
        pcl::io::savePCDFileASCII("filtered.pcd", *cloud_projected);
        std::cout << "Projected point cloud saved to 'filtered.pcd'" << std::endl;
    }

    // 可视化

    // 创建可视化器
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Plane Projection Comparison"));
    viewer->setBackgroundColor(0.05, 0.05, 0.15); // 深蓝色背景

    // 左侧视口 - 原始点云
    int v1(0);
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->setBackgroundColor(0.1, 0.1, 0.2, v1);

    // 添加原始点云
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_red(cloud, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_red, "original_cloud", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud", v1);

    // 添加文本说明
    viewer->addText("Original Point Cloud", 10, 20, 16, 1, 1, 1, "original_text", v1);
    std::string original_count = "Points: " + std::to_string(cloud->size());
    viewer->addText(original_count, 10, 40, 14, 1, 1, 1, "original_count", v1);

    // 右侧视口 - 投影点云
    int v2(0);
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viewer->setBackgroundColor(0.1, 0.2, 0.1, v2);

    // 添加投影点云
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> projected_green(cloud_projected, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_projected, projected_green, "projected_cloud", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "projected_cloud", v2);

    // 添加文本说明
    viewer->addText("Projected Point Cloud", 10, 20, 16, 1, 1, 1, "projected_text", v2);
    std::string projected_count = "Points: " + std::to_string(cloud_projected->size());
    viewer->addText(projected_count, 10, 40, 14, 1, 1, 1, "projected_count", v2);

    // 添加投影平面信息
    std::string plane_info = "Projection Plane: x + y + z = 0";
    viewer->addText(plane_info, 10, 60, 14, 1, 1, 1, "plane_info", v2);

    // 公共设置

    // 添加标题
    viewer->addText("Plane Projection Filter", 300, 20, 18, 1, 1, 1, "title");

    // 添加坐标轴
    viewer->addCoordinateSystem(1.0, "axis_v1", v1);
    viewer->addCoordinateSystem(1.0, "axis_v2", v2);

    // 设置相机参数
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 5, 0, 0, 0, 0, 1, 0);

    std::cout << "\n=== Visualization Started ===" << std::endl;
    std::cout << "Left: Original point cloud (Red)" << std::endl;
    std::cout << "Right: Projected point cloud (Green)" << std::endl;
    std::cout << "Projection plane: x + y + z = 0" << std::endl;
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