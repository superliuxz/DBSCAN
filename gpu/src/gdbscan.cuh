//
// Created by will on 2020-03-23.
//

#ifndef DBSCAN_GDBSCAN_CUH
#define DBSCAN_GDBSCAN_CUH

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "DBSCAN/membership.h"
#include "DBSCAN/utils.h"

namespace GDBSCAN {
int const BLOCK_SIZE = 512;

class Solver {
 public:
  Solver(const std::string &, uint32_t, float);
  /*!
   * Sort the input by the l1norm of each point.
   */
  void sort_input_by_l1norm();
  /*!
   * Spend k*V time, calculate the number of neighbours for each vertex.
   * Returns the number of neighbours of last vertex.
   */
  void calc_num_neighbours();
  /*!
   * Prefix sum.
   */
  void calc_start_pos();
  /*!
   * Spend k*V time, populate the actual neighbours for each vertex.
   */
  void append_neighbours();
  /*!
   * ID the Core and non-Core vertices.
   */
  void identify_cores();
  /*!
   * Identify all the existing clusters using BFS.
   */
  void identify_clusters();

 public:
  std::vector<int> cluster_ids;
  std::vector<DBSCAN::membership> memberships;

#if defined(DBSCAN_TESTING)
 public:
  std::vector<uint32_t> num_neighbours, start_pos;
  std::vector<uint32_t, DBSCAN::utils::NonConstructAllocator<uint32_t>>
      neighbours;
#endif

 private:
  cudaMemcpyKind D2H = cudaMemcpyDeviceToHost;
  cudaMemcpyKind H2D = cudaMemcpyHostToDevice;
  // query params
  float radius_;
  uint32_t num_vtx_{};
  uint32_t total_num_nbs_{};
  uint32_t min_pts_;
  // data structures
  std::vector<float> x_, y_, l1norm_;
  // maps the sorted indices of each vertex to the original index.
  std::vector<uint32_t> vtx_mapper_;
  // gpu vars. Class members to avoid unnecessary copy.
  int num_blocks_{};
  float *dev_x_{}, *dev_y_{}, *dev_l1norm_{};
  uint32_t *dev_vtx_mapper_{}, *dev_num_neighbours_{}, *dev_start_pos_{},
      *dev_neighbours_{};
  DBSCAN::membership *dev_membership_{};
  /*!
   * BFS CPU kernel, which invokes the BFS GPU kernel, and syncs at each level.
   * @param u - starting vertex of BFS.
   * @param cluster - current cluster id.
   */
  void bfs(uint32_t u, int cluster);
};
}  // namespace GDBSCAN

#endif  // DBSCAN_GDBSCAN_CUH