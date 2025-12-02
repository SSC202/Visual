#include <iostream>
#include <thread>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ia_kfpcs.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/time.h>

int main(int argc, char **argv)
{
    pcl::StopWatch timer;

    // 定义点云类型和对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr coarse_aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr final_aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);

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

    std::cout << "=== 4PCS + ICP Point Cloud Registration ===" << std::endl;
    std::cout << "Source point cloud: " << source_cloud->size() << " points" << std::endl;
    std::cout << "Target point cloud: " << target_cloud->size() << " points" << std::endl;

    // 4PCS 粗配准
    std::cout << "\n=== Step 1: Performing 4PCS Coarse Registration ===" << std::endl;

    // 创建4PCS配准对象
    pcl::registration::KFPCSInitialAlignment<pcl::PointXYZ, pcl::PointXYZ> kfpcs;

    // 设置输入源点云和目标点云
    kfpcs.setInputSource(source_cloud);
    kfpcs.setInputTarget(target_cloud);

    // 设置4PCS算法参数
    kfpcs.setApproxOverlap(0.55);   // 源和目标之间的近似重叠度
    kfpcs.setLambda(0.3);           // 平移矩阵的加权系数
    kfpcs.setDelta(0.05);           // 配准后源点云和目标点云之间的距离
    kfpcs.setNumberOfThreads(8);    // 使用8个线程加速
    kfpcs.setNumberOfSamples(1000); // 配准时要使用的随机采样点数量

    std::cout << "4PCS Parameters: ApproxOverlap=0.55, Delta=0.05, NumberOfSamples=1000" << std::endl;

    // 执行4PCS配准
    auto start_time_coarse = std::chrono::steady_clock::now();
    kfpcs.align(*coarse_aligned_cloud);
    auto end_time_coarse = std::chrono::steady_clock::now();
    auto duration_coarse = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_coarse - start_time_coarse);

    // 检查4PCS配准是否成功
    if (!kfpcs.hasConverged())
    {
        std::cout << "\n4PCS coarse registration failed!" << std::endl;
        return -1;
    }

    std::cout << "4PCS registration completed in: " << duration_coarse.count() << " ms" << std::endl;
    std::cout << "4PCS fitness score: " << kfpcs.getFitnessScore() << std::endl;
    std::cout << "4PCS transformation matrix:" << std::endl;
    std::cout << kfpcs.getFinalTransformation() << std::endl;

    // 将粗配准变换应用于源点云
    pcl::transformPointCloud(*source_cloud, *coarse_aligned_cloud, kfpcs.getFinalTransformation());

    // ICP 精配准
    std::cout << "\n=== Step 2: Performing ICP Fine Registration ===" << std::endl;

    // 创建ICP配准对象
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

    // 设置ICP算法参数
    icp.setInputSource(coarse_aligned_cloud);  // 输入粗配准后的点云
    icp.setInputTarget(target_cloud);          // 目标点云
    icp.setTransformationEpsilon(1e-10);       // 为终止条件设置最小转换差异
    icp.setMaxCorrespondenceDistance(1e-2);    // 设置对应点对之间的最大距离
    icp.setEuclideanFitnessEpsilon(1e-10);     // 设置收敛条件是均方误差和小于阈值
    icp.setMaximumIterations(200);             // 最大迭代次数
    icp.setUseReciprocalCorrespondences(true); // 使用相互对应关系

    std::cout << "ICP Parameters: MaxCorrespondenceDistance=1e-2, EuclideanFitnessEpsilon=1e-10" << std::endl;
    std::cout << "MaximumIterations=200, TransformationEpsilon=1e-10" << std::endl;

    // 执行ICP配准
    auto start_time_fine = std::chrono::steady_clock::now();
    icp.align(*final_aligned_cloud, kfpcs.getFinalTransformation()); // 使用4PCS的结果作为初始变换
    auto end_time_fine = std::chrono::steady_clock::now();
    auto duration_fine = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_fine - start_time_fine);

    // 检查ICP配准是否成功
    if (!icp.hasConverged())
    {
        std::cout << "\nICP fine registration failed!" << std::endl;
        return -1;
    }

    std::cout << "ICP registration completed in: " << duration_fine.count() << " ms" << std::endl;
    std::cout << "ICP fitness score: " << icp.getFitnessScore() << std::endl;
    std::cout << "ICP transformation matrix:" << std::endl;
    std::cout << icp.getFinalTransformation() << std::endl;

    // 计算总变换矩阵（4PCS + ICP）
    Eigen::Matrix4f total_transform = icp.getFinalTransformation();
    std::cout << "\n=== Final Combined Transformation Matrix ===" << std::endl;
    std::cout << total_transform << std::endl;

    // 将最终变换应用于原始源点云
    pcl::transformPointCloud(*source_cloud, *final_aligned_cloud, total_transform);

    // 计算总时间和总配准误差
    auto total_duration = duration_coarse + duration_fine;
    std::cout << "\n=== Registration Summary ===" << std::endl;
    std::cout << "Total registration time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "4PCS time: " << duration_coarse.count() << " ms" << std::endl;
    std::cout << "ICP time: " << duration_fine.count() << " ms" << std::endl;
    std::cout << "Final fitness score: " << icp.getFitnessScore() << std::endl;

    // 质量评估
    if (icp.getFitnessScore() < 0.005f)
    {
        std::cout << "Registration quality: Excellent" << std::endl;
    }
    else if (icp.getFitnessScore() < 0.01f)
    {
        std::cout << "Registration quality: Very Good" << std::endl;
    }
    else if (icp.getFitnessScore() < 0.02f)
    {
        std::cout << "Registration quality: Good" << std::endl;
    }
    else if (icp.getFitnessScore() < 0.05f)
    {
        std::cout << "Registration quality: Acceptable" << std::endl;
    }
    else
    {
        std::cout << "Registration quality: Poor - consider parameter adjustment" << std::endl;
    }

    // 保存配准后的点云
    pcl::io::savePCDFileBinary("aligned_cloud.pcd", *final_aligned_cloud);
    std::cout << "Final aligned point cloud saved to 'aligned_cloud.pcd'" << std::endl;

    // 可视化
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("4PCS + ICP Registration Results"));
    viewer->setBackgroundColor(0.05, 0.05, 0.15); // 深蓝色背景

    // 左侧视口 - 原始点云
    int v1(0);
    viewer->createViewPort(0.0, 0.0, 0.25, 1.0, v1);
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
    viewer->addText("Original Clouds", 10, 20, 16, 1, 1, 1, "original_text", v1);
    viewer->addText("Red: Source", 10, 40, 14, 1, 0, 0, "source_text", v1);
    viewer->addText("Green: Target", 10, 60, 14, 0, 1, 0, "target_text", v1);

    // 中间左侧视口 - 4PCS粗配准结果
    int v2(0);
    viewer->createViewPort(0.25, 0.0, 0.5, 1.0, v2);
    viewer->setBackgroundColor(0.1, 0.15, 0.1, v2);

    // 添加目标点云
    viewer->addPointCloud<pcl::PointXYZ>(target_cloud, target_green, "target_cloud_v2", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target_cloud_v2", v2);

    // 添加4PCS配准后的点云
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> coarse_yellow(coarse_aligned_cloud, 255, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(coarse_aligned_cloud, coarse_yellow, "coarse_aligned_cloud", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "coarse_aligned_cloud", v2);

    // 添加文本说明
    viewer->addText("4PCS Coarse Registration", 10, 20, 16, 1, 1, 1, "coarse_text", v2);
    viewer->addText("Green: Target", 10, 40, 14, 0, 1, 0, "target_text_v2", v2);
    viewer->addText("Yellow: Coarse Aligned", 10, 60, 14, 1, 1, 0, "coarse_text_v2", v2);
    std::string coarse_score = "4PCS Score: " + std::to_string(kfpcs.getFitnessScore());
    viewer->addText(coarse_score, 10, 80, 14, 1, 1, 1, "coarse_score_text", v2);

    // 中间右侧视口 - ICP精配准结果
    int v3(0);
    viewer->createViewPort(0.5, 0.0, 0.75, 1.0, v3);
    viewer->setBackgroundColor(0.15, 0.1, 0.1, v3);

    // 添加目标点云
    viewer->addPointCloud<pcl::PointXYZ>(target_cloud, target_green, "target_cloud_v3", v3);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target_cloud_v3", v3);

    // 添加最终配准后的点云
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> final_blue(final_aligned_cloud, 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ>(final_aligned_cloud, final_blue, "final_aligned_cloud", v3);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "final_aligned_cloud", v3);

    // 添加文本说明
    viewer->addText("ICP Fine Registration", 10, 20, 16, 1, 1, 1, "fine_text", v3);
    viewer->addText("Green: Target", 10, 40, 14, 0, 1, 0, "target_text_v3", v3);
    viewer->addText("Blue: Final Aligned", 10, 60, 14, 0, 0, 1, "final_text_v3", v3);
    std::string final_score = "ICP Score: " + std::to_string(icp.getFitnessScore());
    viewer->addText(final_score, 10, 80, 14, 1, 1, 1, "final_score_text", v3);

    // 右侧视口 - 最终重叠效果
    int v4(0);
    viewer->createViewPort(0.75, 0.0, 1.0, 1.0, v4);
    viewer->setBackgroundColor(0.1, 0.1, 0.1, v4);

    // 添加目标点云
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_white(target_cloud, 255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZ>(target_cloud, target_white, "target_cloud_v4", v4);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target_cloud_v4", v4);

    // 添加最终配准后的点云
    viewer->addPointCloud<pcl::PointXYZ>(final_aligned_cloud, final_blue, "final_aligned_cloud_v4", v4);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "final_aligned_cloud_v4", v4);

    // 添加文本说明
    viewer->addText("Final Overlap View", 10, 20, 16, 1, 1, 1, "overlap_text", v4);
    viewer->addText("White: Target", 10, 40, 14, 1, 1, 1, "target_text_v4", v4);
    viewer->addText("Blue: Final Aligned", 10, 60, 14, 0, 0, 1, "final_text_v4", v4);

    std::string total_time = "Total Time: " + std::to_string(total_duration.count()) + " ms";
    std::string registration_info = "4PCS+ICP Registration";
    viewer->addText(total_time, 10, 80, 14, 1, 1, 1, "time_text", v4);
    viewer->addText(registration_info, 10, 100, 14, 1, 1, 1, "reg_info_text", v4);

    // 公共设置
    viewer->addText("4PCS Coarse + ICP Fine Registration", 300, 20, 18, 1, 1, 1, "title");

    // 添加坐标系
    viewer->addCoordinateSystem(0.5, "axis_v1", v1);
    viewer->addCoordinateSystem(0.5, "axis_v2", v2);
    viewer->addCoordinateSystem(0.5, "axis_v3", v3);
    viewer->addCoordinateSystem(0.5, "axis_v4", v4);

    // 设置相机参数
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 5, 0, 0, 0, 0, 1, 0);

    std::cout << "\n=== Visualization Started ===" << std::endl;
    std::cout << "1st View: Original clouds (Red: Source, Green: Target)" << std::endl;
    std::cout << "2nd View: 4PCS coarse result (Yellow: Coarse aligned)" << std::endl;
    std::cout << "3rd View: ICP fine result (Blue: Final aligned)" << std::endl;
    std::cout << "4th View: Final overlap (White: Target, Blue: Aligned)" << std::endl;
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