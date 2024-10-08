#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>

int main(int argc, char** argv) {
    // Read in the cloud data
    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(
        new pcl::PointCloud<pcl::PointXYZ>);
    reader.read("test.pcd", *cloud);
    std::cout << "PointCloud before filtering has: " << cloud->points.size() << " data points." << std::endl; //*

    // Create the filtering object: downsample the dataset using a leaf size of 1cm
    // 执行降采样滤波，叶子大小1cm
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.01f, 0.01f, 0.01f);
    vg.filter(*cloud_filtered);
    std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size() << " data points."
        << std::endl; //*

    // Create the segmentation object for the planar model and set all the parameters
    // 创建平面模型分割器并初始化参数
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PCDWriter writer;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.02);

    int i = 0, nr_points = (int)cloud_filtered->points.size();
    while (cloud_filtered->points.size() > 0.3 * nr_points) {
        // Segment the largest planar component from the remaining cloud
        // 移除剩余点云中最大的平面
        seg.setInputCloud(cloud_filtered);
        // 执行分割，将分割出来的平面点云索引保存到inliers中
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size() == 0) {
            std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
            break;
        }

        // Extract the planar inliers from the input cloud
        // 从输入点云中取出平面内点
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_filtered);
        extract.setIndices(inliers);
        extract.setNegative(false);

        // Get the points associated with the planar surface
        // 得到与平面相关的点cloud_plane
        extract.filter(*cloud_plane);
        std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size() << " data points."
            << std::endl;

        // Remove the planar inliers, extract the rest
        // 从点云中剔除这些平面内点，提取出剩下的点保存到cloud_f中，并重新赋值给cloud_filtered。
        extract.setNegative(true);
        extract.filter(*cloud_f);
        *cloud_filtered = *cloud_f;
    }

    // Creating the KdTree object for the search method of the extraction
    // 为提取算法的搜索方法创建一个KdTree对象
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_filtered);

    /**
     * 在这里，我们创建一个PointIndices的vector，该vector在vector <int>中包含实际的索引信息。
     * 每个检测到的簇的索引都保存在这里-请注意，cluster_indices是一个vector，包含多个检测到的簇的PointIndices的实例。
     * 因此，cluster_indices[0]包含我们点云中第一个 cluster(簇)的所有索引。
     *
     * 从点云中提取簇（集群）,并将点云索引保存在 cluster_indices 中。
     */
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.02);               // 设置临近搜索的搜索半径（搜索容差）为2cm
    ec.setMinClusterSize(100);                  // 每个簇（集群）的最小大小
    ec.setMaxClusterSize(25000);                // 每个簇（集群）的最大大小
    ec.setSearchMethod(tree);                   // 设置点云搜索算法
    ec.setInputCloud(cloud_filtered);           // 设置输入点云
    ec.extract(cluster_indices);                // 设置提取到的簇，将每个簇以索引的形式保存到cluster_indices;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

    // 为了从点云索引向量中分割出每个簇，必须迭代访问点云索引，
    int j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin();
        it != cluster_indices.end(); ++it) {

        // 每次创建一个新的点云数据集，并且将所有当前簇的点写入到点云数据集中。
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        const std::vector<int>& indices = it->indices;

        for (std::vector<int>::const_iterator pit = indices.begin(); pit != indices.end(); ++pit)
            cloud_cluster->points.push_back(cloud_filtered->points[*pit]);

        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size() << " data points."
            << std::endl;
        /*
            std::stringstream ss;
            ss << "cloud_cluster_" << j << ".pcd";
            writer.write<pcl::PointXYZ>(ss.str(), *cloud_cluster, false); //
        */
        std::stringstream ss;
        ss << "cloud_cluster_" << j;
        // Generate a random (bright) color
        pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> single_color(cloud_cluster);
        viewer->addPointCloud<pcl::PointXYZ>(cloud_cluster, single_color, ss.str());
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, ss.str());

        j++;
    }
    std::cout << "cloud size: " << cluster_indices.size() << std::endl;

    viewer->addCoordinateSystem(0.5);
    while (!viewer->wasStopped()) {
        viewer->spinOnce();
    }

    return (0);
}