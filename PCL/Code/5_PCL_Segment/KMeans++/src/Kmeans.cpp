#include "Kmeans.h"
#include <random>
#include <numeric>
#include <algorithm>
#include <pcl/common/centroid.h>
#include <pcl/common/distances.h>
#include <pcl/common/io.h>

void KMeans::extract(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, std::vector<pcl::Indices> &cluster_idx)
{
	// K-means++ 初始化：使用最远点采样选取初始聚类中心
	std::vector<int> selected_indices;
	selected_indices.reserve(m_clusterNum);
	const size_t num_points = cloud->size();

	// 如果点云为空，直接返回
	if (num_points == 0)
		return;

	// 初始化距离数组
	std::vector<float> distances(num_points, std::numeric_limits<float>::max());

	// 随机选择第一个中心点
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, num_points - 1);
	size_t farthest_index = dis(gen);

	// K-means++ 初始化过程
	for (size_t i = 0; i < m_clusterNum; i++)
	{
		selected_indices.push_back(farthest_index);
		const pcl::PointXYZ &selected = cloud->points[farthest_index];
		double max_dist = 0;

		// 更新所有点到最近中心的距离，并找到最远的点
		for (size_t j = 0; j < num_points; j++)
		{
			float dist = (cloud->points[j].getVector3fMap() - selected.getVector3fMap()).squaredNorm();
			distances[j] = std::min(distances[j], dist);
			if (distances[j] > max_dist)
			{
				max_dist = distances[j];
				farthest_index = j;
			}
		}
	}

	// 获取聚类中心点
	pcl::PointCloud<pcl::PointXYZ>::Ptr centers(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::copyPointCloud(*cloud, selected_indices, *centers);

	// 进行 K-means 聚类迭代
	if (!cloud->empty() && !centers->empty())
	{
		int iterations = 0;
		double center_diff = 0.2;				   // 初始中心点变化量
		const double convergence_threshold = 0.02; // 收敛阈值

		// 迭代直到达到最大迭代次数或中心点变化小于阈值
		while (iterations < m_maxIteration && center_diff > convergence_threshold)
		{
			center_diff = 0;
			cluster_idx.clear();
			cluster_idx.resize(m_clusterNum);

			// 分配每个点到最近的聚类中心
			for (size_t i = 0; i < cloud->points.size(); ++i)
			{
				std::vector<double> distances_to_centers;
				for (size_t j = 0; j < m_clusterNum; ++j)
				{
					// 计算点到聚类中心的欧式距离
					distances_to_centers.emplace_back(pcl::euclideanDistance(cloud->points[i], centers->points[j]));
				}

				// 找到最近的聚类中心
				auto min_dist = std::min_element(distances_to_centers.cbegin(), distances_to_centers.cend());
				int cluster_id = std::distance(distances_to_centers.cbegin(), min_dist);

				// 将点分配到对应的聚类
				cluster_idx[cluster_id].push_back(i);
			}

			// 重新计算聚类中心
			pcl::PointCloud<pcl::PointXYZ> new_centers;
			for (size_t k = 0; k < m_clusterNum; ++k)
			{
				// 如果聚类为空，使用随机点作为新中心
				if (cluster_idx[k].empty())
				{
					int random_index = dis(gen);
					new_centers.points.push_back(cloud->points[random_index]);
					continue;
				}

				// 计算聚类的重心
				Eigen::Vector4f centroid;
				pcl::compute3DCentroid(*cloud, cluster_idx[k], centroid);
				pcl::PointXYZ center{centroid[0], centroid[1], centroid[2]};
				new_centers.points.push_back(center);
			}

			// 计算聚类中心的变化量
			for (size_t s = 0; s < m_clusterNum; ++s)
			{
				center_diff += pcl::euclideanDistance(new_centers.points[s], centers->points[s]);
			}

			// 更新聚类中心
			centers->points.clear();
			*centers = new_centers;

			++iterations;
		}

		std::cout << "K-means++ converged after " << iterations << " iterations" << std::endl;
	}
}