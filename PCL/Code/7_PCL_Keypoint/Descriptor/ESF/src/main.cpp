#include <iostream>
#include <thread>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/esf.h>
#include <pcl/search/kdtree.h>

using namespace std;

/**
 * @brief 计算点云的ESF特征描述符
 * @param object 输入点云
 * @return ESF特征描述符（640维向量）
 */
pcl::PointCloud<pcl::ESFSignature640>::Ptr compute_esf_descriptor(pcl::PointCloud<pcl::PointXYZ>::Ptr &object)
{
    // 存储ESF描述符的对象
    pcl::PointCloud<pcl::ESFSignature640>::Ptr descriptor(new pcl::PointCloud<pcl::ESFSignature640>);

    // ESF估计对象
    pcl::ESFEstimation<pcl::PointXYZ, pcl::ESFSignature640> esf;
    esf.setInputCloud(object);

    // ESF不需要设置KSearch，它使用整个点云
    esf.compute(*descriptor);

    return descriptor;
}

int main()
{
    cout << "ESF (Ensemble of Shape Functions) Feature Extraction" << endl;
    cout << "=====================================================" << endl;

    auto total_start_time = std::chrono::steady_clock::now();

    // 加载源点云
    auto load_start_time = std::chrono::steady_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
    string filename = "cloud.pcd";

    cout << "Loading point cloud from: " << filename << endl;
    pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *source);

    auto load_end_time = std::chrono::steady_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end_time - load_start_time);

    cout << "Point Cloud Information:" << endl;
    cout << "  Loaded " << source->size() << " points" << endl;
    cout << "  Loading time: " << load_duration.count() << " ms" << endl;

    // 计算ESF特征描述
    cout << "\nComputing ESF Descriptor:" << endl;
    auto esf_start_time = std::chrono::steady_clock::now();

    auto source_feature = compute_esf_descriptor(source);

    auto esf_end_time = std::chrono::steady_clock::now();
    auto esf_duration = std::chrono::duration_cast<std::chrono::milliseconds>(esf_end_time - esf_start_time);

    cout << "  ESF descriptor computed" << endl;
    cout << "  Processing time: " << esf_duration.count() << " ms" << endl;
    cout << "  Feature dimension: 640" << endl;
    cout << "  Descriptor type: ESFSignature640" << endl;
    cout << "  Note: ESF is a global descriptor for the entire point cloud" << endl;

    // 分析总结
    auto total_end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);

    cout << "\nAnalysis Summary:" << endl;
    cout << "  Total processing time: " << total_duration.count() << " ms" << endl;
    cout << "  Point cloud size: " << source->size() << endl;
    cout << "  ESF descriptor dimension: 640" << endl;
    cout << "  Descriptor type: Global shape descriptor" << endl;

    // 输出部分特征值作为示例
    if (source_feature->size() > 0)
    {
        cout << "\nFirst 10 values of ESF descriptor:" << endl;
        cout << "  ";
        for (int i = 0; i < 10 && i < 640; ++i)
        {
            cout << source_feature->points[0].histogram[i];
            if (i < 9)
                cout << ", ";
        }
        cout << " ... (total 640 values)" << endl;
    }

    return 0;
}