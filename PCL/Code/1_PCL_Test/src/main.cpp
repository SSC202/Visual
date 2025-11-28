#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

int main()
{
    // 创建一个简单的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->width = 100;
    cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);

    for (size_t i = 0; i < cloud->points.size(); ++i)
    {

        
        cloud->points[i].x = 1024 * rand() / (RAND_MAX + 1.0f);
        cloud->points[i].y = 1024 * rand() / (RAND_MAX + 1.0f);
        cloud->points[i].z = 1024 * rand() / (RAND_MAX + 1.0f);
    }

    // 尝试保存点云文件
    pcl::io::savePCDFileASCII("test_output.pcd", *cloud);

    // 尝试可视化点云 (可选，需要pcl_visualization模块)
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    viewer.addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
    viewer.spin();

    return 0;
}