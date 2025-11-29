#include "Kmeans.h"
#include <random>
#include <numeric>
#include <algorithm>
#include <pcl/common/centroid.h>
#include <pcl/common/distances.h>
#include <pcl/common/io.h> // 添加这个头文件以使用copyPointCloud

void KMeans::extract(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
					 std::vector<pcl::Indices> &cluster_idx)
{
	// 检查输入点云是否有效
	if (cloud->empty())
	{
		std::cerr << "Error: Input point cloud is empty!" << std::endl;
		return;
	}

	// 检查聚类数量是否有效
	if (m_clusterNum <= 0 || m_clusterNum > static_cast<int>(cloud->size()))
	{
		std::cerr << "Error: Invalid cluster number!" << std::endl;
		return;
	}

	// 随机选择初始聚类中心
	pcl::Indices indices(cloud->size());
	std::iota(std::begin(indices), std::end(indices), 0);

	std::random_device rd;
	std::mt19937 prng(rd());
	std::shuffle(indices.begin(), indices.end(), prng);
	indices.resize(m_clusterNum);

	// 获取初始聚类中心
	pcl::PointCloud<pcl::PointXYZ>::Ptr centers(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::copyPointCloud(*cloud, indices, *centers); // 使用正确的copyPointCloud函数

	// 执行K-means聚类
	int iterations = 0;
	double center_diff = 0.0;
	const double convergence_threshold = 0.02; // 收敛阈值

	do
	{
		center_diff = 0.0;
		cluster_idx.clear();
		cluster_idx.resize(m_clusterNum);

		// 将每个点分配到最近的聚类
		for (size_t i = 0; i < cloud->points.size(); ++i)
		{
			std::vector<double> distances;
			distances.reserve(m_clusterNum);

			// 计算到所有聚类中心的距离
			for (size_t j = 0; j < m_clusterNum; ++j)
			{
				distances.emplace_back(pcl::euclideanDistance(cloud->points[i], centers->points[j]));
			}

			// 找到最近的聚类
			auto min_dist_it = std::min_element(distances.cbegin(), distances.cend());
			int cluster_id = std::distance(distances.cbegin(), min_dist_it);
			cluster_idx[cluster_id].push_back(i);
		}

		// 重新计算聚类中心
		pcl::PointCloud<pcl::PointXYZ> new_centers;

		for (size_t k = 0; k < m_clusterNum; ++k)
		{
			// 跳过空聚类
			if (cluster_idx[k].empty())
			{
				// 用随机点重新初始化空聚类
				pcl::PointXYZ random_point = cloud->points[rand() % cloud->size()];
				new_centers.points.push_back(random_point);
				continue;
			}

			// 计算聚类的重心
			Eigen::Vector4f centroid;
			pcl::compute3DCentroid(*cloud, cluster_idx[k], centroid);
			pcl::PointXYZ center{centroid[0], centroid[1], centroid[2]};
			new_centers.points.push_back(center);
		}

		// 计算聚类中心的变化
		for (size_t s = 0; s < m_clusterNum; ++s)
		{
			center_diff += pcl::euclideanDistance(new_centers.points[s], centers->points[s]);
		}

		// 更新中心
		centers->points.clear();
		*centers = new_centers;

		++iterations;

		// 打印进度
		std::cout << "Iteration " << iterations << ": center change = " << center_diff << std::endl;

	} while (iterations < m_maxIteration && center_diff > convergence_threshold);

	std::cout << "K-means converged after " << iterations << " iterations" << std::endl;
}