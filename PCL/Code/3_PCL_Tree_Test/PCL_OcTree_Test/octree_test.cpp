#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>           // OcTree相关定义
#include <pcl/visualization/cloud_viewer.h>     // 可视化

#include <iostream>
#include <vector>
#include <ctime>


int main(int argc, char** argv) {

    // 初始化随机数种子
    srand((unsigned int)time(NULL));

    // 随机化创建点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->width = 1000;
    cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        cloud->points[i].x = 1024.0f * rand() / (RAND_MAX + 1.0f);
        cloud->points[i].y = 1024.0f * rand() / (RAND_MAX + 1.0f);
        cloud->points[i].z = 1024.0f * rand() / (RAND_MAX + 1.0f);
    }


    // resolution参数描述了octree叶子leaf节点的最小体素尺寸
    float resolution = 128.0f;
    
    // 创建 OcTree 对象
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);

    // 设置输入点云
    octree.setInputCloud(cloud);

    // 通过点云构建octree
    octree.addPointsFromInputCloud();

    // 创建搜索点
    pcl::PointXYZ searchPoint;
    searchPoint.x = 1024.0f * rand() / (RAND_MAX + 1.0f);
    searchPoint.y = 1024.0f * rand() / (RAND_MAX + 1.0f);
    searchPoint.z = 1024.0f * rand() / (RAND_MAX + 1.0f);

    // 体素近邻搜索
    std::vector<int> pointIdxVec;

    if (octree.voxelSearch(searchPoint, pointIdxVec)) {
        std::cout << "Neighbors within voxel search at (" << searchPoint.x
            << " " << searchPoint.y
            << " " << searchPoint.z << ")"
            << std::endl;

        for (size_t i = 0; i < pointIdxVec.size(); ++i)
            std::cout << "    " << cloud->points[pointIdxVec[i]].x
            << " " << cloud->points[pointIdxVec[i]].y
            << " " << cloud->points[pointIdxVec[i]].z << std::endl;
    }

    // KNN
    int K = 10;

    std::vector<int> pointIdxNKNSearch;
    std::vector<float> pointNKNSquaredDistance;

    std::cout << "K nearest neighbor search at (" << searchPoint.x
        << " " << searchPoint.y
        << " " << searchPoint.z
        << ") with K=" << K << std::endl;

    if (octree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
        for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
            std::cout << "    " << cloud->points[pointIdxNKNSearch[i]].x
            << " " << cloud->points[pointIdxNKNSearch[i]].y
            << " " << cloud->points[pointIdxNKNSearch[i]].z
            << " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
    }

    // 半径搜索
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    float radius = 256.0f * rand() / (RAND_MAX + 1.0f);

    std::cout << "Neighbors within radius search at (" << searchPoint.x
        << " " << searchPoint.y
        << " " << searchPoint.z
        << ") with radius=" << radius << std::endl;


    if (octree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
        for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
            std::cout << "    " << cloud->points[pointIdxRadiusSearch[i]].x
            << " " << cloud->points[pointIdxRadiusSearch[i]].y
            << " " << cloud->points[pointIdxRadiusSearch[i]].z
            << " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
    }

    // 可视化
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    viewer.setBackgroundColor(0.0, 0.0, 0.5);
    viewer.addPointCloud<pcl::PointXYZ>(cloud, "cloud");

    pcl::PointXYZ originPoint(0.0, 0.0, 0.0);
    // 添加从原点到搜索点的线段
    viewer.addLine(originPoint, searchPoint);
    // 添加一个以搜索点为圆心，搜索半径为半径的球体
    viewer.addSphere(searchPoint, radius, "sphere", 0);
    // 添加一个放到200倍后的坐标系
    viewer.addCoordinateSystem(200);

    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}