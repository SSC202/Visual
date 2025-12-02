#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/time.h>

using namespace std;

/**
 * @brief Visualize point cloud with bounding boxes using four viewports
 * 使用四视口可视化点云及其包围盒
 */
void visualize_bounding_boxes(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                              pcl::PointXYZ &minPtAabb, pcl::PointXYZ &maxPtAabb,
                              pcl::PointXYZ &minPtObb, pcl::PointXYZ &maxPtObb,
                              pcl::PointXYZ &posObb, Eigen::Matrix3f &rMatObb,
                              Eigen::Vector3f &centroid,
                              Eigen::Vector3f &maxVec, Eigen::Vector3f &midVec, Eigen::Vector3f &minVec,
                              const std::string &window_title = "Point Cloud Bounding Box Analysis")
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0.05, 0.05, 0.15); // 深蓝色背景

    // 创建四个视口
    int v1(0), v2(1), v3(2), v4(3);
    viewer->createViewPort(0.0, 0.0, 0.5, 0.5, v1); // 左下：原始点云
    viewer->createViewPort(0.5, 0.0, 1.0, 0.5, v2); // 右下：AABB包围盒
    viewer->createViewPort(0.0, 0.5, 0.5, 1.0, v3); // 左上：OBB包围盒
    viewer->createViewPort(0.5, 0.5, 1.0, 1.0, v4); // 右上：OBB特征向量

    // 设置各视口背景色
    viewer->setBackgroundColor(0.1, 0.1, 0.2, v1);   // 深蓝
    viewer->setBackgroundColor(0.15, 0.1, 0.1, v2);  // 深红
    viewer->setBackgroundColor(0.1, 0.15, 0.1, v3);  // 深绿
    viewer->setBackgroundColor(0.15, 0.1, 0.15, v4); // 深紫

    // 颜色处理器
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        cloud_green(cloud, 0, 225, 0); // 点云 - 绿色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        cloud_white(cloud, 255, 255, 255); // 点云 - 白色

    // 原始点云
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_green, "original_cloud", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud", v1);
    viewer->addText("Original Point Cloud", 10, 20, 16, 1, 1, 1, "original_text", v1);
    viewer->addText("Green: Point Cloud", 10, 40, 14, 0, 1, 0, "cloud_text", v1);

    std::string cloud_info = "Points: " + std::to_string(cloud->size());
    std::string file_info = "File: lamppost.pcd";
    viewer->addText(cloud_info, 10, 60, 12, 1, 1, 1, "cloud_info", v1);
    viewer->addText(file_info, 10, 75, 12, 1, 1, 1, "file_info", v1);

    // 添加坐标系
    viewer->addCoordinateSystem(0.2, "axis_v1", v1);

    // AABB包围盒
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_white, "cloud_aabb", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_aabb", v2);

    // 添加AABB包围盒（红色，半透明）
    viewer->addCube(minPtAabb.x, maxPtAabb.x, minPtAabb.y, maxPtAabb.y, minPtAabb.z, maxPtAabb.z,
                    1.0, 0.0, 0.0, "AABB", v2);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "AABB", v2);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "AABB", v2);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                                        pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "AABB", v2);

    viewer->addText("Axis-Aligned Bounding Box (AABB)", 10, 20, 16, 1, 1, 1, "aabb_title", v2);
    viewer->addText("Red: AABB Bounding Box", 10, 40, 14, 1, 0, 0, "aabb_color", v2);

    std::string aabb_size = "Size: " + std::to_string(maxPtAabb.x - minPtAabb.x).substr(0, 5) + " x " +
                            std::to_string(maxPtAabb.y - minPtAabb.y).substr(0, 5) + " x " +
                            std::to_string(maxPtAabb.z - minPtAabb.z).substr(0, 5);
    viewer->addText(aabb_size, 10, 60, 12, 1, 1, 1, "aabb_size", v2);

    viewer->addCoordinateSystem(0.2, "axis_v2", v2);

    // OBB包围盒
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_white, "cloud_obb", v3);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_obb", v3);

    // 添加OBB包围盒（蓝色，半透明）
    Eigen::Vector3f position(posObb.x, posObb.y, posObb.z);
    Eigen::Quaternionf quat(rMatObb);
    viewer->addCube(position, quat, maxPtObb.x - minPtObb.x, maxPtObb.y - minPtObb.y, maxPtObb.z - minPtObb.z,
                    "OBB", v3);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "OBB", v3);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "OBB", v3);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "OBB", v3);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                                        pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "OBB", v3);

    viewer->addText("Oriented Bounding Box (OBB)", 10, 20, 16, 1, 1, 1, "obb_title", v3);
    viewer->addText("Blue: OBB Bounding Box", 10, 40, 14, 0, 0, 1, "obb_color", v3);

    std::string obb_size = "Size: " + std::to_string(maxPtObb.x - minPtObb.x).substr(0, 5) + " x " +
                           std::to_string(maxPtObb.y - minPtObb.y).substr(0, 5) + " x " +
                           std::to_string(maxPtObb.z - minPtObb.z).substr(0, 5);
    viewer->addText(obb_size, 10, 60, 12, 1, 1, 1, "obb_size", v3);

    viewer->addCoordinateSystem(0.2, "axis_v3", v3);

    // OBB特征向量
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_white, "cloud_vectors", v4);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_vectors", v4);

    // 添加质心
    pcl::PointXYZ center(centroid(0), centroid(1), centroid(2));
    viewer->addSphere(center, 0.05, 1, 1, 0, "centroid", v4);

    // 添加OBB特征向量
    pcl::PointXYZ x_axis(maxVec(0) * 0.3 + centroid(0), maxVec(1) * 0.3 + centroid(1), maxVec(2) * 0.3 + centroid(2));
    pcl::PointXYZ y_axis(midVec(0) * 0.3 + centroid(0), midVec(1) * 0.3 + centroid(1), midVec(2) * 0.3 + centroid(2));
    pcl::PointXYZ z_axis(minVec(0) * 0.3 + centroid(0), minVec(1) * 0.3 + centroid(1), minVec(2) * 0.3 + centroid(2));

    viewer->addLine(center, x_axis, 1.0f, 0.0f, 0.0f, "major_eigen_vector", v4);
    viewer->addLine(center, y_axis, 0.0f, 1.0f, 0.0f, "middle_eigen_vector", v4);
    viewer->addLine(center, z_axis, 0.0f, 0.0f, 1.0f, "minor_eigen_vector", v4);

    viewer->addText("OBB Eigenvectors", 10, 20, 16, 1, 1, 1, "vectors_title", v4);
    viewer->addText("Red: Major Eigenvector", 10, 40, 14, 1, 0, 0, "major_vector", v4);
    viewer->addText("Green: Middle Eigenvector", 10, 55, 14, 0, 1, 0, "middle_vector", v4);
    viewer->addText("Blue: Minor Eigenvector", 10, 70, 14, 0, 0, 1, "minor_vector", v4);
    viewer->addText("Yellow: Centroid", 10, 85, 14, 1, 1, 0, "centroid_text", v4);

    viewer->addCoordinateSystem(0.2, "axis_v4", v4);

    // 设置相机参数
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 30, 0, 0, 0, 0, 1, 0);

    // 主循环
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main()
{
    std::cout << "=== Point Cloud Bounding Box Analysis ===" << std::endl;
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

    // 计算AABB包围盒
    std::cout << "\n=== Step 1: Computing AABB Bounding Box ===" << std::endl;
    auto aabb_start_time = std::chrono::steady_clock::now();

    pcl::PointXYZ minPtAabb, maxPtAabb;
    pcl::getMinMax3D(*cloud, minPtAabb, maxPtAabb);

    auto aabb_end_time = std::chrono::steady_clock::now();
    auto aabb_duration = std::chrono::duration_cast<std::chrono::milliseconds>(aabb_end_time - aabb_start_time);

    std::cout << "AABB Min Point: (" << minPtAabb.x << ", " << minPtAabb.y << ", " << minPtAabb.z << ")" << std::endl;
    std::cout << "AABB Max Point: (" << maxPtAabb.x << ", " << maxPtAabb.y << ", " << maxPtAabb.z << ")" << std::endl;
    std::cout << "AABB Dimensions: " << maxPtAabb.x - minPtAabb.x << " x "
              << maxPtAabb.y - minPtAabb.y << " x "
              << maxPtAabb.z - minPtAabb.z << std::endl;
    std::cout << "AABB computation time: " << aabb_duration.count() << " ms" << std::endl;

    // 计算OBB包围盒
    std::cout << "\n=== Step 2: Computing OBB Bounding Box ===" << std::endl;
    auto obb_start_time = std::chrono::steady_clock::now();

    pcl::MomentOfInertiaEstimation<pcl::PointXYZ> mie;
    mie.setInputCloud(cloud);
    mie.compute();

    float maxValue, midValue, minValue;       // 三个特征值
    Eigen::Vector3f maxVec, midVec, minVec;   // 特征值对应的特征向量
    Eigen::Vector3f centroid;                 // 点云质心
    pcl::PointXYZ minPtObb, maxPtObb, posObb; // OBB包围盒最小值、最大值以及位姿
    Eigen::Matrix3f rMatObb;                  // OBB包围盒对应的旋转矩阵

    mie.getOBB(minPtObb, maxPtObb, posObb, rMatObb);  // 获取OBB对应的相关参数
    mie.getEigenValues(maxValue, midValue, minValue); // 获取特征值
    mie.getEigenVectors(maxVec, midVec, minVec);      // 获取特征向量
    mie.getMassCenter(centroid);                      // 获取点云中心坐标

    auto obb_end_time = std::chrono::steady_clock::now();
    auto obb_duration = std::chrono::duration_cast<std::chrono::milliseconds>(obb_end_time - obb_start_time);

    std::cout << "OBB Min Point: (" << minPtObb.x << ", " << minPtObb.y << ", " << minPtObb.z << ")" << std::endl;
    std::cout << "OBB Max Point: (" << maxPtObb.x << ", " << maxPtObb.y << ", " << maxPtObb.z << ")" << std::endl;
    std::cout << "OBB Dimensions: " << maxPtObb.x - minPtObb.x << " x "
              << maxPtObb.y - minPtObb.y << " x "
              << maxPtObb.z - minPtObb.z << std::endl;
    std::cout << "Centroid: (" << centroid(0) << ", " << centroid(1) << ", " << centroid(2) << ")" << std::endl;
    std::cout << "Eigenvalues: " << maxValue << " (max), " << midValue << " (mid), " << minValue << " (min)" << std::endl;
    std::cout << "OBB computation time: " << obb_duration.count() << " ms" << std::endl;

    // 计算包围盒体积比
    double aabb_volume = (maxPtAabb.x - minPtAabb.x) * (maxPtAabb.y - minPtAabb.y) * (maxPtAabb.z - minPtAabb.z);
    double obb_volume = (maxPtObb.x - minPtObb.x) * (maxPtObb.y - minPtObb.y) * (maxPtObb.z - minPtObb.z);
    double volume_ratio = obb_volume / aabb_volume * 100.0;

    std::cout << "\n=== Step 3: Volume Analysis ===" << std::endl;
    std::cout << "AABB Volume: " << aabb_volume << std::endl;
    std::cout << "OBB Volume: " << obb_volume << std::endl;
    std::cout << "OBB/AABB Volume Ratio: " << volume_ratio << "%" << std::endl;

    if (volume_ratio < 100.0)
    {
        std::cout << "OBB is " << 100.0 - volume_ratio << "% smaller than AABB (better fit)" << std::endl;
    }
    else
    {
        std::cout << "OBB is " << volume_ratio - 100.0 << "% larger than AABB" << std::endl;
    }

    // 计算总时间并输出
    auto total_end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);

    std::cout << "\n=== Analysis Summary ===" << std::endl;
    std::cout << "AABB computation time: " << aabb_duration.count() << " ms" << std::endl;
    std::cout << "OBB computation time: " << obb_duration.count() << " ms" << std::endl;
    std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;

    // 调用可视化函数
    visualize_bounding_boxes(cloud, minPtAabb, maxPtAabb, minPtObb, maxPtObb,
                             posObb, rMatObb, centroid, maxVec, midVec, minVec,
                             "Point Cloud Bounding Box Analysis");

    return 0;
}