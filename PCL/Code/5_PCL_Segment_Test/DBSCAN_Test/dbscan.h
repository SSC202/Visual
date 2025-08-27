#pragma once
#include<pcl/point_types.h>
#include <pcl/search/kdtree.h>

template<typename PointT>bool
dbscan(const pcl::PointCloud<PointT>& cloud_in, std::vector<pcl::Indices>& cluster_idx, const double& epsilon, const int& minpts)
{
	std::vector<bool> cloud_processed(cloud_in.size(), false);

	for (size_t i = 0; i < cloud_in.size(); ++i)
	{
		if (cloud_processed[i] != false)
		{
			continue;
		}
		pcl::Indices seed_queue;
		// �����ڵ�����Ƿ����minpts,�жϸõ��Ƿ�Ϊ���Ķ���
		auto tree = std::make_shared<pcl::search::KdTree<PointT>>();
		tree->setInputCloud(cloud_in.makeShared());
		pcl::Indices k_indices;
		std::vector<float> k_distances;
		if (tree->radiusSearch(cloud_in.points[i], epsilon, k_indices, k_distances) >= minpts)
		{
			seed_queue.push_back(i);
			cloud_processed[i] = true;
		}
		else
		{
			continue;
		}

		int seed_index = 0;
		while (seed_index < seed_queue.size())
		{
			pcl::Indices indices;
			std::vector<float> dists;
			if (tree->radiusSearch(cloud_in.points[seed_queue[seed_index]], epsilon, indices, dists) < minpts)//��������ֵΪ��������
			{
				//������С��minpts�ĵ�����Ǳ߽�㡢��㡢Ҳ�����Ǵص�һ���֣������Ϊ�Ѵ���matlab���ǵ��������
				++seed_index;
				continue;
			}
			for (size_t j = 0; j < indices.size(); ++j)
			{
				if (cloud_processed[indices[j]])
				{
					continue;
				}
				seed_queue.push_back(indices[j]);
				cloud_processed[indices[j]] = true;
			}
			++seed_index;
		}

		cluster_idx.push_back(seed_queue);

	}

	if (cluster_idx.size() == 0)
		return false;
	else
		return true;
}


