//
// Created by William Liu on 2020-01-24.
//

#ifndef GDBSCAN_INCLUDE_SOLVER_H_
#define GDBSCAN_INCLUDE_SOLVER_H_

#include <fstream>
#include <memory>

#include "Dataset.h"
#include "Dimension.h"
#include "Graph.h"
#include "spdlog/spdlog.h"

namespace GDBSCAN {

template<class DimensionType>
class Solver {
 public:
  explicit Solver(std::unique_ptr<std::ifstream> in,
                  size_t num_nodes,
                  int min_pts,
                  double radius) :
      num_nodes_(num_nodes), min_pts_(min_pts), radius_(radius) {
    ifs_ = std::move(in);
    logger_ = spdlog::get("console");
  }
#ifdef TESTING
  const std::vector<DimensionType> dataset_view() const {
    return dataset_->view();
  }
  const Graph graph_view() const {
    return *graph_;
  }
#endif
  void prepare_dataset() {
    dataset_ = std::make_unique<Dataset<DimensionType>>(num_nodes_);
    size_t n;
    float x, y;
    if (std::is_same_v<DimensionType, dimension::TwoD>) {
      while (*ifs_ >> n >> x >> y) {
        (*dataset_)[n] = DimensionType(x, y);
      }
    } else {
      throw std::runtime_error("DimensionType not supported!");
    }
    ifs_->close();
  }

  /*
   * Algorithm 1 in Andrade et al.
   */
  void make_graph() {
    if (ifs_->is_open()) {
      throw std::runtime_error(
          "Input file stream still open (should not happen)!");
    }
    if (dataset_ == nullptr) {
      throw std::runtime_error("Call prepare_dataset to generate the dataset!");
    }
    graph_ = std::make_unique<Graph>(num_nodes_);
    for (size_t u = 0; u < num_nodes_; ++u) {
      for (size_t v = u + 1; v < num_nodes_; ++v) {
        if ((*dataset_)[u] - (*dataset_)[v] <= radius_) {
          graph_->insert_edge(u, v);
        }
      }
    }
    graph_->finalize();
    classify_nodes();
  }

  /*
   * Algorithm 2 (BFS) in Andrade et al.
   */
  void identify_cluster() const {
    int cluster = 1;
    for (size_t node = 0; node < num_nodes_; ++node) {
      if (graph_->cluster_ids[node] == -1
          && graph_->membership[node] == membership::Core) {
        graph_->cluster_ids[node] = cluster;
        logger_->debug("start bfs on node {} with cluster {}", node, cluster);
        bfs(node, cluster);
        ++cluster;
      }
    }
  }

 private:
  size_t num_nodes_;
  int min_pts_;
  double radius_;
  std::unique_ptr<Dataset < DimensionType>> dataset_ = nullptr;
  std::unique_ptr<Graph> graph_ = nullptr;
  std::unique_ptr<std::ifstream> ifs_ = nullptr;
  std::shared_ptr<spdlog::logger> logger_ = nullptr;

  /*
   * Classify nodes to Core or Noise; the Border nodes are classified in the BFS
   * stage (Algorithm 2 in Andrade et al).
   */
  void classify_nodes() const {
    if (graph_ == nullptr) {
      throw std::runtime_error("Call make_graph to generate the graph!");
    }
    for (size_t node = 0; node < num_nodes_; ++node) {
      if (graph_->Va[node * 2] >= min_pts_) {
        graph_->membership[node] = membership::Core;
      } else {
        graph_->membership[node] = membership::Noise;
      }
    }
  }

  /*
   * BFS. Start from |node| and visit all the reachable neighbours. If a
   * neighbour is Noise, relabel it to Border.
   */
  void bfs(size_t node, int cluster) const {
    std::vector<size_t> q{node};
    std::vector<size_t> next_level;
    while (!q.empty()) {
      for (size_t &curr : q) {
        logger_->debug("visiting node {}", curr);
        // Relabel a reachable Noise node, but do not keep exploring.
        if (graph_->membership[curr] == membership::Noise) {
          logger_->debug("\tnode {} is relabeled from Noise to Border", curr);
          graph_->membership[curr] = membership::Border;
          continue;
        }

        size_t num_neighbours = graph_->Va[2 * curr];
        size_t start_pos = graph_->Va[2 * curr + 1];

        for (size_t i = 0; i < num_neighbours; ++i) {
          size_t nb = graph_->Ea[start_pos + i];
          if (graph_->cluster_ids[nb] == -1) {
            // cluster the node
            logger_->debug("\tnode {} is clustered tp {}", nb, cluster);
            graph_->cluster_ids[nb] = cluster;
            logger_->debug("\tneighbour {} of node {} is queued", nb, curr);
            next_level.emplace_back(nb);
          }
        }
      }
      q = next_level;
      next_level.clear();
    }
  }
};

template<class DimensionType>
static std::unique_ptr<Solver<DimensionType>> make_solver(std::string input,
                                                          int min_pts,
                                                          double radius) {
  size_t num_nodes;
  auto ifs = std::make_unique<std::ifstream>(input);
  *ifs >> num_nodes;
  return std::make_unique<Solver<DimensionType>>(std::move(ifs),
                                                 num_nodes,
                                                 min_pts,
                                                 radius);
}
}

#endif //GDBSCAN_INCLUDE_SOLVER_H_
