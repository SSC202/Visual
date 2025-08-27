#include <iostream>
#include <thread>

#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std::chrono_literals;

pcl::visualization::PCLVisualizer::Ptr simpleVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, pcl::PointCloud<pcl::PointXYZ>::ConstPtr final = nullptr) 
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");

    if (final != nullptr) {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(final, 255, 0, 0);
        viewer->addPointCloud<pcl::PointXYZ>(final, color_handler, "final cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "final cloud");
    }

    viewer->addCoordinateSystem(1.0, "global");
    viewer->initCameraParameters();
    return (viewer);
    }
/**
 * ʹ�÷�����
 *
 * random_sample_consensus     ���������ⲿ���ƽ��
 * random_sample_consensus -f  ���������ⲿ���ƽ�棬������ƽ���ڲ���
 *
 * random_sample_consensus -s  ���������ⲿ�������
 * random_sample_consensus -sf ���������ⲿ������壬�����������ڲ���
 */
int main(int argc, char** argv) {

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr final(new pcl::PointCloud<pcl::PointXYZ>);

    // ������õ���
    cloud->width = 500;
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->points.resize(cloud->width * cloud->height);
    for (std::size_t i = 0; i < cloud->points.size(); ++i) {
        if (pcl::console::find_argument(argc, argv, "-s") >= 0 || pcl::console::find_argument(argc, argv, "-sf") >= 0) {
            cloud->points[i].x = 1024 * rand() / (RAND_MAX + 1.0);
            cloud->points[i].y = 1024 * rand() / (RAND_MAX + 1.0);
            if (i % 5 == 0)     // ���ܻ�ɢ����������
                cloud->points[i].z = 1024 * rand() / (RAND_MAX + 1.0);
            else if (i % 2 == 0)// ��������������
                cloud->points[i].z = sqrt(1 - (cloud->points[i].x * cloud->points[i].x)
                    - (cloud->points[i].y * cloud->points[i].y));
            else // �����帺������
                cloud->points[i].z = -sqrt(1 - (cloud->points[i].x * cloud->points[i].x)
                    - (cloud->points[i].y * cloud->points[i].y));
        }
        else {
            cloud->points[i].x = 1024 * rand() / (RAND_MAX + 1.0);
            cloud->points[i].y = 1024 * rand() / (RAND_MAX + 1.0);
            if (i % 2 == 0)
                cloud->points[i].z = 1024 * rand() / (RAND_MAX + 1.0);
            else
                cloud->points[i].z = -1 * (cloud->points[i].x + cloud->points[i].y);
        }
    }

    std::vector<int> inliers;

    // RANSAC ���ģ�ͽ���
    // Բ��
    pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr
        model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZ>(cloud));
    // ƽ��
    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr
        model_p(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud));

    // RANSAC 
    if (pcl::console::find_argument(argc, argv, "-f") >= 0) {
        pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);
        ransac.setDistanceThreshold(.01);
        ransac.computeModel();      // ִ�����
        ransac.getInliers(inliers);
    }
    else if (pcl::console::find_argument(argc, argv, "-sf") >= 0) {
        pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_s);
        ransac.setDistanceThreshold(.01);
        ransac.computeModel();      // ִ�����
        ransac.getInliers(inliers);
    }

    // ��cloud��ָ�������ĵ㿽����final������
    pcl::copyPointCloud(*cloud, inliers, *final);

    // ���ӻ�
    pcl::visualization::PCLVisualizer::Ptr viewer;
    if (pcl::console::find_argument(argc, argv, "-f") >= 0 || pcl::console::find_argument(argc, argv, "-sf") >= 0)
        viewer = simpleVis(cloud, final);
    else
        viewer = simpleVis(cloud);
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(100ms);
    }
    return 0;
}
