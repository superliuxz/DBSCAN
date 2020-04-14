//
// Created by William Liu on 2020-01-24.
//

#ifndef DBSCAN_INCLUDE_SOLVER_H_
#define DBSCAN_INCLUDE_SOLVER_H_

#include <immintrin.h>
#include <nmmintrin.h>

#include <fstream>
#include <memory>
#include <thread>

#include "dataset.h"
#include "graph.h"
#include "spdlog/spdlog.h"

namespace DBSCAN {

class Solver {
 public:
  std::vector<int> cluster_ids;
  std::vector<DBSCAN::membership> memberships;
  explicit Solver(const std::string&, const uint64_t&, const float&,
                  const uint8_t&);
  void sort_data_by_l1norm();
  /*
   * For each two vertices, if the distance is <= |squared_radius_|, insert them
   * into the graph (|temp_adj_|). Part of Algorithm 1 (Andrade et al).
   */
  void insert_edges();
  /*
   * Construct |Va| and |Ea| from |temp_adj|. Part of Algorithm 1
   * (Andrade et al).
   */
  void finalize_graph() const {
    using namespace std::chrono;
    high_resolution_clock::time_point start = high_resolution_clock::now();
    graph_->finalize();
    duration<double> time_spent =
        duration_cast<duration<double>>(high_resolution_clock::now() - start);
    logger_->info("finalize_graph takes {} seconds", time_spent.count());
  }
  /*
   * Classify vertices to Core or Noise; the Border vertices are classified in
   * the BFS stage. Algorithm 2 from Andrade et al.
   */
  void classify_vertices();
  /*
   * Initiate a BFS on each un-clustered vertex. Algorithm 2 from Andrade et al.
   */
  void identify_cluster();

 private:
  uint64_t num_vtx_{}, min_pts_;
  float radius_, squared_radius_;
  uint8_t num_threads_;
  std::vector<uint64_t> vtx_mapper_;
  std::shared_ptr<spdlog::logger> logger_ = nullptr;
  /*
   * Start from |vertex| and visit all the reachable neighbours. If a neighbour
   * is Noise, relabel it to Border.
   */
  void bfs_(uint64_t, int);

#if defined(AVX)
  const float max_radius_ = std::sqrt(std::numeric_limits<float>::max()) - 1;
  __m256 sq_rad8_;
#endif
#if defined(DBSCAN_TESTING)
 public:
#else
 private:
#endif
  std::unique_ptr<DBSCAN::input_type::TwoDimPoints> dataset_ = nullptr;
  std::unique_ptr<Graph> graph_ = nullptr;
};
}  // namespace DBSCAN

#endif  // DBSCAN_INCLUDE_SOLVER_H_
