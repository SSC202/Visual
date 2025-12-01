#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <iostream>
#include <thread>
#include <chrono>

int main()
{
    // 定义点云类型和对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
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

    std::cout << "=== SAC-IA Point Cloud Registration ===" << std::endl;
    std::cout << "Source point cloud: " << source_cloud->size() << " points" << std::endl;
    std::cout << "Target point cloud: " << target_cloud->size() << " points" << std::endl;

    // 点云下采样
    std::cout << "\nDownsampling point clouds..." << std::endl;
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(0.5f, 0.5f, 0.5f);

    voxel_filter.setInputCloud(source_cloud);
    voxel_filter.filter(*source_downsampled);

    voxel_filter.setInputCloud(target_cloud);
    voxel_filter.filter(*target_downsampled);

    std::cout << "Downsampled source: " << source_downsampled->size() << " points" << std::endl;
    std::cout << "Downsampled target: " << target_downsampled->size() << " points" << std::endl;

    // 创建KD树
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kd_tree(new pcl::search::KdTree<pcl::PointXYZ>);

    // 计算法向量
    std::cout << "\nComputing normals..." << std::endl;
    pcl::PointCloud<pcl::Normal>::Ptr source_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr target_normals(new pcl::PointCloud<pcl::Normal>);

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(kd_tree);
    normal_estimator.setKSearch(10);

    normal_estimator.setInputCloud(source_downsampled);
    normal_estimator.compute(*source_normals);

    normal_estimator.setInputCloud(target_downsampled);
    normal_estimator.compute(*target_normals);

    // 计算FPFH特征
    std::cout << "Computing FPFH features..." << std::endl;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_features(new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_features(new pcl::PointCloud<pcl::FPFHSignature33>);

    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_estimator;
    fpfh_estimator.setSearchMethod(kd_tree);
    fpfh_estimator.setKSearch(15);

    fpfh_estimator.setInputCloud(source_downsampled);
    fpfh_estimator.setInputNormals(source_normals);
    fpfh_estimator.compute(*source_features);

    fpfh_estimator.setInputCloud(target_downsampled);
    fpfh_estimator.setInputNormals(target_normals);
    fpfh_estimator.compute(*target_features);

    std::cout << "Source features: " << source_features->size() << " descriptors" << std::endl;
    std::cout << "Target features: " << target_features->size() << " descriptors" << std::endl;

    // 创建 SAC-IA 配准对象
    pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;

    // 设置输入源点云和目标点云
    sac_ia.setInputSource(source_downsampled);
    sac_ia.setSourceFeatures(source_features);
    sac_ia.setInputTarget(target_downsampled);
    sac_ia.setTargetFeatures(target_features);

    // 设置 SAC-IA 算法参数
    sac_ia.setMinSampleDistance(0.05f);        // 样本间最小距离
    sac_ia.setMaxCorrespondenceDistance(0.1f); // 最大对应距离
    sac_ia.setMaximumIterations(1000);         // 最大迭代次数
    sac_ia.setNumberOfSamples(3);              // 使用的样本数量
    sac_ia.setCorrespondenceRandomness(15);    // 使用的邻域数量

    std::cout << "\n=== Performing SAC-IA Registration ===" << std::endl;
    std::cout << "Parameters: MinSampleDistance=0.05, MaxCorrespondenceDistance=0.1" << std::endl;
    std::cout << "MaximumIterations=1000, NumberOfSamples=3" << std::endl;

    // 执行配准
    auto start_time = std::chrono::steady_clock::now();
    sac_ia.align(*aligned_cloud);
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Registration completed in: " << duration.count() << " ms" << std::endl;

    // 输出配准结果
    if (sac_ia.hasConverged())
    {
        std::cout << "\nSAC-IA registration successful!" << std::endl;
        std::cout << "Fitness score: " << sac_ia.getFitnessScore() << std::endl;
        std::cout << "Transformation matrix:" << std::endl;
        std::cout << sac_ia.getFinalTransformation() << std::endl;

        // 使用变换矩阵对原始源点云进行变换
        pcl::transformPointCloud(*source_cloud, *aligned_cloud, sac_ia.getFinalTransformation());

        // 保存配准后的点云
        pcl::io::savePCDFile<pcl::PointXYZ>("aligned_cloud.pcd", *aligned_cloud);
        std::cout << "Aligned point cloud saved to 'aligned_cloud.pcd'" << std::endl;

        // 质量评估
        if (sac_ia.getFitnessScore() < 0.05f)
        {
            std::cout << "Registration quality: Excellent" << std::endl;
        }
        else if (sac_ia.getFitnessScore() < 0.1f)
        {
            std::cout << "Registration quality: Good" << std::endl;
        }
        else if (sac_ia.getFitnessScore() < 0.2f)
        {
            std::cout << "Registration quality: Acceptable" << std::endl;
        }
        else
        {
            std::cout << "Registration quality: Poor - consider parameter adjustment" << std::endl;
        }
    }
    else
    {
        std::cout << "\nSAC-IA registration failed!" << std::endl;
        return -1;
    }

    // 可视化
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("SAC-IA Point Cloud Registration"));
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

    // 添加目标点云（白色）
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_white(target_cloud, 255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZ>(target_cloud, target_white, "target_cloud_v3", v3);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target_cloud_v3", v3);

    // 添加配准后的点云（蓝色）
    viewer->addPointCloud<pcl::PointXYZ>(aligned_cloud, aligned_blue, "aligned_cloud_v3", v3);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "aligned_cloud_v3", v3);

    // 添加文本说明
    viewer->addText("Overlap View", 10, 20, 16, 1, 1, 1, "overlap_text", v3);
    viewer->addText("White: Target cloud", 10, 40, 14, 1, 1, 1, "target_text_v3", v3);
    viewer->addText("Blue: Aligned cloud", 10, 60, 14, 0, 0, 1, "aligned_text_v3", v3);
    std::string fitness_info = "Fitness: " + std::to_string(sac_ia.getFitnessScore());
    viewer->addText(fitness_info, 10, 80, 14, 1, 1, 1, "fitness_text", v3);

    // 公共设置
    viewer->addText("SAC-IA Point Cloud Registration", 400, 20, 18, 1, 1, 1, "title");

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