#pragma once
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <vector>
#include <memory>

/**
 * @brief DBSCAN density-based clustering algorithm
 *
 * @tparam PointT Point type
 * @param cloud_in Input point cloud
 * @param cluster_idx Output cluster indices
 * @param epsilon Neighborhood search radius
 * @param minpts Minimum number of points required to form a cluster
 * @return true Clustering successful
 * @return false Clustering failed
 */
template <typename PointT>
bool dbscan(const pcl::PointCloud<PointT> &cloud_in,
			std::vector<pcl::Indices> &cluster_idx,
			const double &epsilon,
			const int &minpts)
{
	// Initialize processed flags for all points
	std::vector<bool> cloud_processed(cloud_in.size(), false);

	// Create KD-tree for efficient neighborhood search
	auto tree = std::make_shared<pcl::search::KdTree<PointT>>();
	tree->setInputCloud(cloud_in.makeShared());

	// Process each point in the cloud
	for (size_t i = 0; i < cloud_in.size(); ++i)
	{
		// Skip already processed points
		if (cloud_processed[i])
		{
			continue;
		}

		pcl::Indices seed_queue;
		pcl::Indices k_indices;
		std::vector<float> k_distances;

		// Check if current point is a core point (has enough neighbors within epsilon)
		if (tree->radiusSearch(cloud_in.points[i], epsilon, k_indices, k_distances) >= minpts)
		{
			// Mark as core point and add to seed queue
			seed_queue.push_back(i);
			cloud_processed[i] = true;
		}
		else
		{
			// Point is noise or border point, skip for now
			continue;
		}

		// Expand the cluster using seed points
		int seed_index = 0;
		while (seed_index < seed_queue.size())
		{
			pcl::Indices indices;
			std::vector<float> dists;

			// Find neighbors of current seed point
			if (tree->radiusSearch(cloud_in.points[seed_queue[seed_index]], epsilon, indices, dists) < minpts)
			{
				// Not a core point, but may be border point
				++seed_index;
				continue;
			}

			// Process all neighbors
			for (size_t j = 0; j < indices.size(); ++j)
			{
				if (cloud_processed[indices[j]])
				{
					continue; // Already processed
				}

				// Add to seed queue and mark as processed
				seed_queue.push_back(indices[j]);
				cloud_processed[indices[j]] = true;
			}

			++seed_index;
		}

		// Save the completed cluster
		cluster_idx.push_back(seed_queue);
	}

	// Return success status
	return !cluster_idx.empty();
}