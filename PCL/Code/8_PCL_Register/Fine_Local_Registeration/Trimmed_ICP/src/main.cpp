#include <iostream>
#include <thread>
#include <chrono>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ia_kfpcs.h>                // 4PCS算法头文件
#include <pcl/recognition/ransac_based/trimmed_icp.h> // Trimmed ICP头文件
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/time.h>

using namespace std;

/**
 * @brief Visualize registration results with four viewports
 * 使用四视口可视化配准结果
 */
void visualize_registration(pcl::PointCloud<pcl::PointXYZ>::Ptr &source,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr &target,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr &coarse_result,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr &icp_result,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr &trimmed_result,
                            const std::string &window_title = "4PCS + ICP + Trimmed ICP Registration Results")
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0.05, 0.05, 0.15); // 深蓝色背景

    // 创建四个视口
    int v1(0), v2(1), v3(2), v4(3);
    viewer->createViewPort(0.0, 0.0, 0.25, 1.0, v1); // 原始点云
    viewer->createViewPort(0.25, 0.0, 0.5, 1.0, v2); // 4PCS粗配准结果
    viewer->createViewPort(0.5, 0.0, 0.75, 1.0, v3); // ICP精配准结果
    viewer->createViewPort(0.75, 0.0, 1.0, 1.0, v4); // Trimmed ICP最终结果

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
        coarse_yellow(coarse_result, 255, 255, 0); // 4PCS结果 - 黄色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        icp_green(icp_result, 0, 255, 0); // ICP结果 - 绿色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        trimmed_magenta(trimmed_result, 255, 0, 255); // Trimmed结果 - 洋红色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        target_white(target, 255, 255, 255); // 目标点云 - 白色（用于重叠视图）

    // 视口1: 原始点云
    viewer->addPointCloud<pcl::PointXYZ>(source, source_red, "source_cloud", v1);
    viewer->addPointCloud<pcl::PointXYZ>(target, target_blue, "target_cloud", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source_cloud", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target_cloud", v1);
    viewer->addText("Original Point Clouds", 10, 20, 16, 1, 1, 1, "original_text", v1);
    viewer->addText("Red: Source Cloud", 10, 40, 14, 1, 0, 0, "source_text", v1);
    viewer->addText("Blue: Target Cloud", 10, 60, 14, 0, 0, 1, "target_text", v1);

    // 视口2: 4PCS粗配准结果
    viewer->addPointCloud<pcl::PointXYZ>(target, target_blue, "target_cloud_v2", v2);
    viewer->addPointCloud<pcl::PointXYZ>(coarse_result, coarse_yellow, "coarse_result_cloud", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target_cloud_v2", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "coarse_result_cloud", v2);
    viewer->addText("4PCS Coarse Registration", 10, 20, 16, 1, 1, 1, "coarse_text", v2);
    viewer->addText("Blue: Target", 10, 40, 14, 0, 0, 1, "target_text_v2", v2);
    viewer->addText("Yellow: Coarse Result", 10, 60, 14, 1, 1, 0, "coarse_result_text", v2);

    // 视口3: ICP精配准结果
    viewer->addPointCloud<pcl::PointXYZ>(target, target_blue, "target_cloud_v3", v3);
    viewer->addPointCloud<pcl::PointXYZ>(icp_result, icp_green, "icp_result_cloud", v3);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target_cloud_v3", v3);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "icp_result_cloud", v3);
    viewer->addText("ICP Fine Registration", 10, 20, 16, 1, 1, 1, "icp_text", v3);
    viewer->addText("Blue: Target", 10, 40, 14, 0, 0, 1, "target_text_v3", v3);
    viewer->addText("Green: ICP Result", 10, 60, 14, 0, 1, 0, "icp_result_text", v3);

    // 视口4: Trimmed ICP最终结果（重叠视图）
    viewer->addPointCloud<pcl::PointXYZ>(target, target_white, "target_cloud_v4", v4);
    viewer->addPointCloud<pcl::PointXYZ>(trimmed_result, trimmed_magenta, "trimmed_result_cloud", v4);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target_cloud_v4", v4);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "trimmed_result_cloud", v4);
    viewer->addText("Trimmed ICP Final Result", 10, 20, 16, 1, 1, 1, "trimmed_text", v4);
    viewer->addText("White: Target", 10, 40, 14, 1, 1, 1, "target_text_v4", v4);
    viewer->addText("Magenta: Trimmed Result", 10, 60, 14, 1, 0, 1, "trimmed_result_text", v4);

    // 添加坐标系
    viewer->addCoordinateSystem(0.1, "axis_v1", v1);
    viewer->addCoordinateSystem(0.1, "axis_v2", v2);
    viewer->addCoordinateSystem(0.1, "axis_v3", v3);
    viewer->addCoordinateSystem(0.1, "axis_v4", v4);

    // 设置相机参数
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 5, 0, 0, 0, 0, 1, 0);

    // 主循环
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main(int argc, char *argv[])
{
    std::cout << "=== 4PCS + ICP + Trimmed ICP Registration ===" << std::endl;
    std::cout << "PCL Version: 1.15.0" << std::endl;

    // 总计时开始
    auto total_start_time = std::chrono::steady_clock::now();

    // 加载源点云
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

    std::cout << "\n=== Point Cloud Information ===" << std::endl;
    std::cout << "Source point cloud: " << source_cloud->size() << " points" << std::endl;
    std::cout << "Target point cloud: " << target_cloud->size() << " points" << std::endl;

    // 4PCS粗配准
    std::cout << "\n=== Step 1: Performing 4PCS Coarse Registration ===" << std::endl;

    auto p4pcs_start_time = std::chrono::steady_clock::now();

    // 创建4PCS配准对象
    pcl::registration::KFPCSInitialAlignment<pcl::PointXYZ, pcl::PointXYZ> kfpcs;

    // 设置输入源点云和目标点云
    kfpcs.setInputSource(source_cloud);
    kfpcs.setInputTarget(target_cloud);

    // 设置4PCS算法参数
    kfpcs.setApproxOverlap(0.55);   // 源和目标之间的近似重叠度
    kfpcs.setLambda(0.3);           // 平移矩阵的加权系数
    kfpcs.setDelta(0.05);           // 配准后源点云和目标点云之间的距离
    kfpcs.setNumberOfThreads(10);   // 使用8个线程加速
    kfpcs.setNumberOfSamples(1000); // 配准时要使用的随机采样点数量

    std::cout << "4PCS Parameters:" << std::endl;
    std::cout << "  ApproxOverlap: 0.55" << std::endl;
    std::cout << "  Delta: 0.05" << std::endl;
    std::cout << "  Lambda: 0.3" << std::endl;
    std::cout << "  NumberOfSamples: 1000" << std::endl;
    std::cout << "  NumberOfThreads: 10" << std::endl;

    // 执行4PCS配准
    pcl::PointCloud<pcl::PointXYZ>::Ptr coarse_aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    kfpcs.align(*coarse_aligned_cloud);

    auto p4pcs_end_time = std::chrono::steady_clock::now();
    auto p4pcs_duration = std::chrono::duration_cast<std::chrono::milliseconds>(p4pcs_end_time - p4pcs_start_time);

    // 检查4PCS配准是否成功
    if (!kfpcs.hasConverged())
    {
        std::cerr << "\n4PCS coarse registration failed!" << std::endl;
        return -1;
    }

    std::cout << "4PCS registration completed in: " << p4pcs_duration.count() << " ms" << std::endl;
    std::cout << "4PCS fitness score: " << kfpcs.getFitnessScore() << std::endl;

    // 获取4PCS变换矩阵
    Eigen::Matrix4f p4pcs_transformation = kfpcs.getFinalTransformation();
    std::cout << "4PCS transformation matrix:" << std::endl;
    std::cout << p4pcs_transformation << std::endl;

    // 生成4PCS结果点云用于可视化
    pcl::PointCloud<pcl::PointXYZ>::Ptr p4pcs_visualization(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*source_cloud, *p4pcs_visualization, p4pcs_transformation);

    // 传统ICP精配准
    std::cout << "\n=== Step 2: Performing Traditional ICP Fine Registration ===" << std::endl;

    auto icp_start_time = std::chrono::steady_clock::now();

    pcl::PointCloud<pcl::PointXYZ>::Ptr icp_result(new pcl::PointCloud<pcl::PointXYZ>);

    int iterations = 35;
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(source_cloud);     // 使用原始源点云
    icp.setInputTarget(target_cloud);     // 目标点云
    icp.setMaximumIterations(iterations); // 设置最大迭代次数
    icp.setMaxCorrespondenceDistance(15); // 设置最大对应点距离
    icp.setTransformationEpsilon(1e-10);  // 设置精度
    icp.setEuclideanFitnessEpsilon(0.01); // 设置收敛条件

    std::cout << "ICP Parameters:" << std::endl;
    std::cout << "  MaximumIterations: " << iterations << std::endl;
    std::cout << "  MaxCorrespondenceDistance: 15" << std::endl;
    std::cout << "  TransformationEpsilon: 1e-10" << std::endl;
    std::cout << "  EuclideanFitnessEpsilon: 0.01" << std::endl;

    // 使用4PCS结果作为ICP的初始变换
    icp.align(*icp_result, p4pcs_transformation);

    auto icp_end_time = std::chrono::steady_clock::now();
    auto icp_duration = std::chrono::duration_cast<std::chrono::milliseconds>(icp_end_time - icp_start_time);

    if (icp.hasConverged())
    {
        std::cout << "ICP registration completed in: " << icp_duration.count() << " ms" << std::endl;
        std::cout << "ICP fitness score: " << icp.getFitnessScore() << std::endl;

        Eigen::Matrix4f icp_transformation = icp.getFinalTransformation();
        std::cout << "ICP transformation matrix:" << std::endl;
        std::cout << icp_transformation << std::endl;
    }
    else
    {
        std::cerr << "Error: ICP has not converged!" << std::endl;
        return -1;
    }

    // Trimmed-ICP精细化处理
    std::cout << "\n=== Step 3: Performing Trimmed ICP Refinement ===" << std::endl;

    auto trimmed_start_time = std::chrono::steady_clock::now();

    pcl::recognition::TrimmedICP<pcl::PointXYZ, double> trimmed_icp;
    trimmed_icp.init(target_cloud); // 初始化目标点云

    float sigma = 0.96;                     // 设置内点比例
    int Np = icp_result->size();            // 使用ICP结果点云的点数
    int Npo = static_cast<int>(Np * sigma); // 参与配准的点对数量

    trimmed_icp.setNewToOldEnergyRatio(sigma); // 设置内点能量比例

    std::cout << "Trimmed ICP Parameters:" << std::endl;
    std::cout << "  Inlier ratio (sigma): " << sigma << std::endl;
    std::cout << "  Number of point pairs (Npo): " << Npo << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr trimmed_result(new pcl::PointCloud<pcl::PointXYZ>);
    *trimmed_result = *icp_result;

    Eigen::Matrix4d trimmed_transformation = Eigen::Matrix4d::Identity();

    // 执行Trimmed ICP配准
    trimmed_icp.align(*trimmed_result, Npo, trimmed_transformation);

    auto trimmed_end_time = std::chrono::steady_clock::now();
    auto trimmed_duration = std::chrono::duration_cast<std::chrono::milliseconds>(trimmed_end_time - trimmed_start_time);

    std::cout << "Trimmed ICP refinement completed in: " << trimmed_duration.count() << " ms" << std::endl;
    std::cout << "Trimmed result point cloud size: " << trimmed_result->size() << " points" << std::endl;

    // 计算总时间
    auto total_end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);

    // 结果总结
    std::cout << "\n=== Registration Summary ===" << std::endl;
    std::cout << "Total registration time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "  4PCS time: " << p4pcs_duration.count() << " ms" << std::endl;
    std::cout << "  ICP time: " << icp_duration.count() << " ms" << std::endl;
    std::cout << "  Trimmed ICP time: " << trimmed_duration.count() << " ms" << std::endl;
    std::cout << "Final fitness score (ICP): " << icp.getFitnessScore() << std::endl;

    // 质量评估
    float fitness_score = icp.getFitnessScore();
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
    std::cout << "2nd View: 4PCS coarse result (Blue: Target, Yellow: Coarse aligned)" << std::endl;
    std::cout << "3rd View: ICP fine result (Blue: Target, Green: ICP aligned)" << std::endl;
    std::cout << "4th View: Trimmed ICP final result (White: Target, Magenta: Trimmed result)" << std::endl;
    std::cout << "Press 'q' to exit the visualization window" << std::endl;
    std::cout << "Use mouse to rotate and scroll to zoom" << std::endl;

    // 结果可视化
    visualize_registration(source_cloud, target_cloud, p4pcs_visualization,
                           icp_result, trimmed_result,
                           "4PCS + ICP + Trimmed ICP Registration Results");

    return 0;
}