//
// Created by William Liu on 2020-01-24.
//

#ifndef GDBSCAN_INCLUDE_RUNNER_H_
#define GDBSCAN_INCLUDE_RUNNER_H_

#include <fstream>
#include <memory>

#include "Dataset.h"
#include "Dimension.h"
#include "Graph.h"

namespace GDBSCAN {

template<class DimensionType>
class Runner {
 public:
  explicit Runner(size_t num_nodes, int min_pts, double radius) :
      num_nodes_(num_nodes), min_pts_(min_pts), radius_(radius) {
  }
#ifdef TESTING
  const std::vector<DimensionType> dataset_view() const {
    return dataset_->view();
  }
  const Graph graph_view() const {
    return *graph_;
  }
#endif
  void prepare_dataset(std::unique_ptr<std::ifstream> in) {
    dataset_ = std::make_unique<Dataset<DimensionType>>(num_nodes_);
    size_t n;
    float x, y;
    if (std::is_same_v<DimensionType, dimension::TwoD>) {
      while (*in >> n >> x >> y) {
        (*dataset_)[n] = DimensionType(x, y);
      }
    } else {
      throw std::runtime_error("DimensionType not supported!");
    }
  }

  void make_graph() {
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
  }

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

 private:
  size_t num_nodes_;
  int min_pts_;
  double radius_;
  std::unique_ptr<Dataset < DimensionType>> dataset_ = nullptr;
  std::unique_ptr<Graph> graph_ = nullptr;
};
}

#endif //GDBSCAN_INCLUDE_RUNNER_H_
