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
    // ִ�н������˲���Ҷ�Ӵ�С1cm
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.01f, 0.01f, 0.01f);
    vg.filter(*cloud_filtered);
    std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size() << " data points."
        << std::endl; //*

    // Create the segmentation object for the planar model and set all the parameters
    // ����ƽ��ģ�ͷָ�������ʼ������
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
        // �Ƴ�ʣ�����������ƽ��
        seg.setInputCloud(cloud_filtered);
        // ִ�зָ���ָ������ƽ������������浽inliers��
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size() == 0) {
            std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
            break;
        }

        // Extract the planar inliers from the input cloud
        // �����������ȡ��ƽ���ڵ�
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_filtered);
        extract.setIndices(inliers);
        extract.setNegative(false);

        // Get the points associated with the planar surface
        // �õ���ƽ����صĵ�cloud_plane
        extract.filter(*cloud_plane);
        std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size() << " data points."
            << std::endl;

        // Remove the planar inliers, extract the rest
        // �ӵ������޳���Щƽ���ڵ㣬��ȡ��ʣ�µĵ㱣�浽cloud_f�У������¸�ֵ��cloud_filtered��
        extract.setNegative(true);
        extract.filter(*cloud_f);
        *cloud_filtered = *cloud_f;
    }

    // Creating the KdTree object for the search method of the extraction
    // Ϊ��ȡ�㷨��������������һ��KdTree����
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_filtered);

    /**
     * ��������Ǵ���һ��PointIndices��vector����vector��vector <int>�а���ʵ�ʵ�������Ϣ��
     * ÿ����⵽�Ĵص�����������������-��ע�⣬cluster_indices��һ��vector�����������⵽�Ĵص�PointIndices��ʵ����
     * ��ˣ�cluster_indices[0]�������ǵ����е�һ�� cluster(��)������������
     *
     * �ӵ�������ȡ�أ���Ⱥ��,������������������ cluster_indices �С�
     */
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.02);               // �����ٽ������������뾶�������ݲΪ2cm
    ec.setMinClusterSize(100);                  // ÿ���أ���Ⱥ������С��С
    ec.setMaxClusterSize(25000);                // ÿ���أ���Ⱥ��������С
    ec.setSearchMethod(tree);                   // ���õ��������㷨
    ec.setInputCloud(cloud_filtered);           // �����������
    ec.extract(cluster_indices);                // ������ȡ���Ĵأ���ÿ��������������ʽ���浽cluster_indices;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

    // Ϊ�˴ӵ������������зָ��ÿ���أ�����������ʵ���������
    int j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin();
        it != cluster_indices.end(); ++it) {

        // ÿ�δ���һ���µĵ������ݼ������ҽ����е�ǰ�صĵ�д�뵽�������ݼ��С�
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