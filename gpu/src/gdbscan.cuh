//
// Created by will on 2020-03-23.
//

#ifndef DBSCAN_GDBSCAN_CUH
#define DBSCAN_GDBSCAN_CUH

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "membership.h"
#include "utils.h"

namespace GDBSCAN {
int const BLOCK_SIZE = 1024;

class Solver {
 public:
  Solver(const std::string &, const uint64_t &, const float &);
  // Spend n^2 time, calculate the number of neighbours for each vertex.
  void calc_num_neighbours();
  // prefix sum
  void inline calc_start_pos() {
    // It's faster to computer the prefix sum on host because memcpy is avoided.
    thrust::exclusive_scan(thrust::host, num_neighbours_.cbegin(),
                           num_neighbours_.cend(), start_pos_.begin());
  }
  // Spend n^2 time, populate the actual neighbours for each vertex.
  void append_neighbours();
  // ID the Core and non-Core vertices.
  void inline identify_cores() {
    for (uint64_t i = 0; i < num_vtx_; ++i) {
      if (num_neighbours_[i] >= min_pts_)
        memberships[i] = DBSCAN::membership::Core;
    }
  }
  // Identify all the existing clusters using BFS.
  void identify_clusters();

 public:
  std::vector<int> cluster_ids;
  std::vector<DBSCAN::membership> memberships;
#if defined(TESTING)
 public:
#else
 private:
#endif
  // data structures
  std::vector<uint64_t> num_neighbours_, start_pos_;
  std::vector<uint64_t, DBSCAN::utils::NonConstructAllocator<uint64_t>>
      neighbours_;

 private:
  // query params
  float squared_radius_;
  uint64_t num_vtx_{};
  uint64_t min_pts_;
  // data structures
  std::vector<float> x_, y_;
  // gpu vars. Class members to avoid unnecessary copy.
  float *dev_x_{}, *dev_y_{};
  uint64_t *dev_num_neighbours_{}, *dev_start_pos_{}, *dev_neighbours_{};
  void bfs(uint64_t u, int cluster);
};
}  // namespace GDBSCAN

#endif  // DBSCAN_GDBSCAN_CUH