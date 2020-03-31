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
  /*
   * Spend n^2 time, calculate the number of neighbours for each vertex.
   * Returns the number of neighbours of last vertex.
   */
  void calc_num_neighbours();
  /*
   * Prefix sum;
   * Returns the start pos of last vertex.
   */
  void calc_start_pos();
  /*
   * Spend n^2 time, populate the actual neighbours for each vertex.
   */
  void append_neighbours();
  /*
   * ID the Core and non-Core vertices.
   */
  void identify_cores();
  /*
   * Identify all the existing clusters using BFS.
   */
  void identify_clusters();

 public:
  std::vector<int> cluster_ids;
  std::vector<DBSCAN::membership> memberships;

#if GDBSCAN_TESTING == 1
 public:
  std::vector<uint64_t> num_neighbours, start_pos;
  std::vector<uint64_t, DBSCAN::utils::NonConstructAllocator<uint64_t>>
      neighbours;
#endif

 private:
  cudaMemcpyKind D2H = cudaMemcpyDeviceToHost;
  cudaMemcpyKind H2D = cudaMemcpyHostToDevice;
  // query params
  float squared_radius_;
  uint64_t num_vtx_{};
  uint64_t total_num_nbs_{};
  uint64_t min_pts_;
  // data structures
  std::vector<float> x_, y_;
  // gpu vars. Class members to avoid unnecessary copy.
  int num_blocks{};
  float *dev_x_{}, *dev_y_{};
  uint64_t *dev_num_neighbours_{}, *dev_start_pos_{}, *dev_neighbours_{};
  DBSCAN::membership *dev_membership_{};
  /*
   * BFS CPU kernel, which invokes the BFS GPU kernel, and syncs at each level.
   */
  void bfs(uint64_t u, int cluster);
};
}  // namespace GDBSCAN

#endif  // DBSCAN_GDBSCAN_CUH