#include "Kmeans.h"
#include <random>
#include <numeric>			
#include <algorithm>
#include <pcl/common/centroid.h>
#include <pcl/common/distances.h>


void KMeans::extract(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::vector<pcl::Indices>& cluster_idx)
{
	// ���ѡȡ�������ĵ�
	pcl::Indices indices(cloud->size());
	// ��0��ʼ����˳������ķ�ʽ��indices��ֵ
	std::iota(std::begin(indices), std::end(indices), (int)0);
	std::random_device rd;
	std::mt19937 prng(rd());
	// �Ը���Ԫ�ؽ������������
	std::shuffle(indices.begin(), indices.end(), prng);
	indices.resize((int)(m_clusterNum));
	// ��ȡ�������ĵ�
	pcl::PointCloud<pcl::PointXYZ>::Ptr m_center(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::copyPointCloud(*cloud, indices, *m_center);

	// ����KMeans����
	if (!cloud->empty() && !m_center->empty())
	{
		int iterations = 0;
		double sum_diff = 0.2;
		// ������ڵ�������������������֮��С��0.02��ֹͣ
		while (!(iterations >= m_maxIteration || sum_diff <= 0.02))

		{
			sum_diff = 0;
			std::vector<int> points_processed(cloud->points.size(), 0);
			cluster_idx.clear();
			cluster_idx.resize(m_clusterNum);
			for (size_t i = 0; i < cloud->points.size(); ++i)

			{
				if (!points_processed[i])
				{
					std::vector<double>dists(0, 0);
					for (size_t j = 0; j < m_clusterNum; ++j)
					{
						// �������е㵽�������ĵ��ŷʽ����
						dists.emplace_back(pcl::euclideanDistance(cloud->points[i], m_center->points[j]));
					}
					std::vector<double>::const_iterator min_dist = std::min_element(dists.cbegin(), dists.cend());
					int it = std::distance(dists.cbegin(), min_dist); // ��ȡ��Сֵ���ڵ�λ��
					cluster_idx[it].push_back(i);                     // �Ž���С�������ڵĴ�
					points_processed[i] = 1;
				}

				else
					continue;
			}

			// ���¼�������ĵ�
			pcl::PointCloud<pcl::PointXYZ> new_centre;
			for (size_t k = 0; k < m_clusterNum; ++k)
			{
				Eigen::Vector4f centroid;
				pcl::compute3DCentroid(*cloud, cluster_idx.at(k), centroid);
				pcl::PointXYZ center{ centroid[0] ,centroid[1] ,centroid[2] };
				new_centre.points.push_back(center);
			}

			// ����������ĵ�ı仯��
			for (size_t s = 0; s < m_clusterNum; ++s)
			{
				sum_diff += pcl::euclideanDistance(new_centre.points[s], m_center->points[s]);
			}

			m_center->points.clear();
			*m_center = new_centre;

			++iterations;
		}

	}

}

