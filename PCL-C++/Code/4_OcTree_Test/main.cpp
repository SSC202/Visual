#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>

#include <iostream>
#include <vector>
#include <ctime>
#include <pcl/visualization/cloud_viewer.h>

int main(int argc, char **argv) {
    srand((unsigned int) time(NULL));

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // Generate pointcloud data
    cloud->width = 1000;
    cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);

    for (size_t i = 0; i < cloud->points.size(); ++i) {
        cloud->points[i].x = 1024.0f * rand() / (RAND_MAX + 1.0f);
        cloud->points[i].y = 1024.0f * rand() / (RAND_MAX + 1.0f);
        cloud->points[i].z = 1024.0f * rand() / (RAND_MAX + 1.0f);
    }

//    float resolution = 0.01f;
    // 设置分辨率为128
    float resolution = 128.0f;
    // resolution该参数描述了octree叶子leaf节点的最小体素尺寸。
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);
    // 设置输入点云
    octree.setInputCloud(cloud);
    // 通过点云构建octree
    octree.addPointsFromInputCloud();

    pcl::PointXYZ searchPoint;

    searchPoint.x = 1024.0f * rand() / (RAND_MAX + 1.0f);
    searchPoint.y = 1024.0f * rand() / (RAND_MAX + 1.0f);
    searchPoint.z = 1024.0f * rand() / (RAND_MAX + 1.0f);

    // Neighbors within voxel search
    // 方式一：“体素近邻搜索”,它把查询点所在的体素中其他点的索引作为查询结果返回,
    // 结果以点索引向量的形式保存,因此搜索点和搜索结果之间的距离取决于八叉树的分辨率参数
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

    // K nearest neighbor search
    // 方式二：K 近邻搜索,本例中K被设置成10, "K 近邻搜索”方法把搜索结果写到两个分开的向量中,
    // 第一个pointIdxNKNSearch 包含搜索结果〈结果点的索引的向量〉
    // 第二个pointNKNSquaredDistance 保存相应的搜索点和近邻之间的距离平方。
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

    // Neighbors within radius search
    // 方式三：半径内近邻搜索
    // “半径内近邻搜索”原理和“K 近邻搜索”类似,它的搜索结果被写入两个分开的向量中,
    // 这两个向量分别存储结果点的索引和对应的距离平方
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
