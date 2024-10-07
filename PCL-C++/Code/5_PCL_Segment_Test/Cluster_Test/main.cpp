#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>       // ����������ȡ����
#include <pcl/filters/voxel_grid.h>            // �����˲�
#include <pcl/kdtree/kdtree.h>                 // kd��
#include <pcl/sample_consensus/method_types.h> // ��������
#include <pcl/sample_consensus/model_types.h>  // ����ģ��
#include <pcl/ModelCoefficients.h>             // ģ��ϵ��
#include <pcl/segmentation/sac_segmentation.h> // ��������ָ�
#include <pcl/segmentation/extract_clusters.h> // ŷʽ����ָ�
#include <pcl/visualization/pcl_visualizer.h> 

using namespace std;

int main(int argc, char** argv)
{
    //--------------------------��ȡ���泡������---------------------------------
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile<pcl::PointXYZ>("test.pcd", *cloud);
    cout << "��ȡ����: " << cloud->points.size() << " ��." << endl;

    //---------------------------�����˲��²���----------------------------------
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.01f, 0.01f, 0.01f);
    vg.filter(*cloud_filtered);
    cout << "�����˲�����: " << cloud_filtered->points.size() << " ��." << endl;

    //--------------------����ƽ��ģ�ͷָ�Ķ������ò���-----------------------
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);    // �ָ�ģ��,ƽ��ģ��
    seg.setMethodType(pcl::SAC_RANSAC);       // �������Ʒ���,�������һ���ԡ�
    seg.setMaxIterations(100);                // ���ĵ����Ĵ���
    seg.setDistanceThreshold(0.02);           // ���÷���ģ�͵��ڵ���ֵ

    // -------------ģ�ͷָ�,ֱ��ʣ�����������30%����,ȷ��ģ�͵��ƽϺ�----------
    int i = 0, nr_points = (int)cloud_filtered->points.size();// �²���ǰ��������
    while (cloud_filtered->points.size() > 0.3 * nr_points)

    {
        seg.setInputCloud(cloud_filtered);
        seg.segment(*inliers, *coefficients);// �ָ�
        if (inliers->indices.size() == 0)
        {
            cout << "Could not estimate a planar model for the given dataset." << endl;
            break;
        }
        //---------------------------����������ȡ����-------------------------------
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_filtered);
        extract.setIndices(inliers);         // ��ȡ����ƽ��ģ�͵��ڵ�
        extract.setNegative(false);
        //--------------------------ƽ��ģ���ڵ�------------------------------------
        extract.filter(*cloud_plane);
        cout << "ƽ��ģ��: " << cloud_plane->points.size() << "����." << endl;
        //-------------------��ȥƽ����ڵ㣬��ȡʣ�����---------------------------
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
        extract.setNegative(true);
        extract.filter(*cloud_f);
        *cloud_filtered = *cloud_f;         // ʣ�����
    }

    // --------------����ƽ���ϵĵ�����,��ʹ��ŷʽ������㷨�Ե��ƾ���ָ�----------
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_filtered);              // ����ƽ���������ĵ���
    vector<pcl::PointIndices> cluster_indices;        // ����������
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;// ŷʽ�������
    ec.setClusterTolerance(0.02);                     // ���ý��������������뾶Ϊ2cm��Ҳ��������ͬ�����ŵ�֮�����Сŷ�Ͼ��룩
    ec.setMinClusterSize(100);                        // ����һ��������Ҫ�����ٵĵ���ĿΪ100
    ec.setMaxClusterSize(25000);                      // ����һ��������Ҫ��������ĿΪ25000
    ec.setSearchMethod(tree);                         // ���õ��Ƶ���������
    ec.setInputCloud(cloud_filtered);
    ec.extract(cluster_indices);                      // �ӵ�������ȡ���࣬������������������cluster_indices��
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster_all(new pcl::PointCloud<pcl::PointXYZ>);

    //------------�������ʵ�������cluster_indices,ֱ���ָ���о���---------------
    int j = 0;
    for (vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        //�����µĵ������ݼ�cloud_cluster�������е�ǰ����д�뵽�������ݼ���
        for (vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
            cloud_cluster->points.push_back(cloud_filtered->points[*pit]); //��ȡÿһ�������ŵĵ�

        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size() << " data points." << endl;
        stringstream ss;
        ss << "cloud_cluster_" << j << ".pcd";
        pcl::PCDWriter writer;
        writer.write<pcl::PointXYZ>(ss.str(), *cloud_cluster, false);
        j++;

        *cloud_cluster_all += *cloud_cluster;
    }
    pcl::io::savePCDFileASCII("cloud_cluster_all.pcd", *cloud_cluster_all);

    //------------------------������ʾ------------------------------------
    pcl::visualization::PCLVisualizer viewer("3D Viewer");
    viewer.setBackgroundColor(0, 0, 0);
    //viewer.addCoordinateSystem (1.0);
    viewer.initCameraParameters();
    //--------------------ƽ���ϵĵ��ơ���ɫ------------------------------
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_plane_handler(cloud_plane, 255, 0, 0);
    viewer.addPointCloud(cloud_plane, cloud_plane_handler, "plan point");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "plan point");

    //--------------------ƽ����ĵ��ơ���ɫ------------------------------
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_cluster_handler(cloud_cluster_all, 0, 255, 0);
    viewer.addPointCloud(cloud_cluster_all, cloud_cluster_handler, "cloud_cluster point");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_cluster point");

    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }
    return (0);
}
