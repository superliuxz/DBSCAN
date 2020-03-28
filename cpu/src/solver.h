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
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace DBSCAN {

class Solver {
 public:
  explicit Solver(const std::string&, const size_t&, const float&,
                  const uint8_t&);
#if defined(TESTING)
  inline const DBSCAN::input_type::TwoDimPoints& dataset_view() const {
    return *dataset_;
  }
#endif
  [[nodiscard]] const Graph& graph_view() const { return *graph_; }
  /*
   * For each two nodes, if the distance is <= |squared_radius_|, insert them
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
   * Classify nodes to Core or Noise; the Border nodes are classified in the BFS
   * stage. Algorithm 2 from Andrade et al.
   */
  void classify_nodes() const;
  /*
   * Initiate a BFS on each un-clustered node. Algorithm 2 from Andrade et al.
   */
  void identify_cluster() const;

 private:
  size_t num_nodes_{};
  size_t min_pts_;
  float squared_radius_;
#if defined(AVX)
  __m256 sq_rad8_;
#endif
  uint8_t num_threads_;
  std::unique_ptr<DBSCAN::input_type::TwoDimPoints> dataset_ = nullptr;
  std::unique_ptr<Graph> graph_ = nullptr;
  std::shared_ptr<spdlog::logger> logger_ = nullptr;
  /*
   * Start from |node| and visit all the reachable neighbours. If a neighbour
   * is Noise, relabel it to Border.
   */
  void bfs(size_t start_node, int cluster) const;
};
}  // namespace DBSCAN

#endif  // DBSCAN_INCLUDE_SOLVER_H_
