#pragma once

#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class KMeans
{
private:
    int m_maxIteration;  // 最大迭代次数
    int m_clusterNum;    // 聚类数量

public:
    /**
     * @brief KMeans类构造函数
     * @param k 聚类数量
     * @param max_iteration 最大迭代次数
     */
    KMeans(int k, int max_iteration) : 
        m_clusterNum(k), 
        m_maxIteration(max_iteration) 
    {}
    
    ~KMeans() = default;

    /**
     * @brief 使用K-means算法提取聚类
     * @param cloud 输入点云
     * @param cluster_idx 输出聚类索引
     */
    void extract(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
                 std::vector<pcl::Indices>& cluster_idx);
};