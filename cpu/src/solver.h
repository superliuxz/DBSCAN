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
#include "grid.h"
#include "spdlog/spdlog.h"

namespace DBSCAN {

class Solver {
 public:
  std::vector<int> cluster_ids;
  std::vector<DBSCAN::membership> memberships;
  explicit Solver(const std::string&, uint64_t, float, uint8_t);
  /*
   * Construct the search grid. Each cell has range {[x0, x0+eps),[y0, y0+eps)}.
   * The number of vtx of each grid is stored in |grid_vtx_counter_|; the vtx
   * indices reside within each cell is stored in |grid_|.
   */
  inline void ConstructGrid() { grid_->Construct(dataset_->d1, dataset_->d2); }
  /*
   * For each two vertices, if the distance is <= |squared_radius_|, insert them
   * into the graph (|temp_adj_|).
   */
  void InsertEdges();
  /*
   * Construct |num_nbs| and |neighbours| from |temp_adj|.
   */
  void FinalizeGraph() const {
    using namespace std::chrono;
    high_resolution_clock::time_point start = high_resolution_clock::now();
    graph_->Finalize();
    duration<double> time_spent =
        duration_cast<duration<double>>(high_resolution_clock::now() - start);
    logger_->info("FinalizeGraph takes {} seconds", time_spent.count());
  }
  /*
   * Classify vertices to Core or Noise; the Border vertices are classified in
   * the BFS stage.
   */
  void ClassifyNoises();
  /*
   * Initiate a BFS on each un-clustered vertex.
   */
  void IdentifyClusters();

 private:
  uint64_t num_vtx_{}, min_pts_;
  float squared_radius_;
  uint8_t num_threads_;
  std::unique_ptr<Grid> grid_ = nullptr;
  std::shared_ptr<spdlog::logger> logger_ = nullptr;
  /*
   * Start from |vertex| and visit all the reachable neighbours. If a neighbour
   * is Noise, relabel it to Border.
   */
  void BFS_(uint64_t, int);

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
