//
// Created by William Liu on 2020-01-23.
//

#ifndef GDBSCAN_INCLUDE_GRAPH_H_
#define GDBSCAN_INCLUDE_GRAPH_H_

#include <spdlog/spdlog.h>

#include <vector>

#include "membership.h"
#include "utils.h"

namespace GDBSCAN {

class Graph {
 public:
  std::vector<size_t> Va;
  std::vector<size_t, GDBSCAN::utils::NonConstructAllocator<size_t>> Ea;
  std::vector<int> cluster_ids;
  std::vector<membership> memberships;
  // ctor
  explicit Graph(const size_t&, const size_t&);
  // insert edge
#if defined(BIT_ADJ)
  void insert_edge(const size_t&, const size_t&, const uint64_t&);
#else
  void start_insert(const size_t& u) { temp_adj_[u].reserve(num_nodes_); }
  void insert_edge(const size_t&, const size_t&);
  void finish_insert(const size_t& u) { temp_adj_[u].shrink_to_fit(); }
#endif
  // construct Va and Ea.
  void finalize();
  // set |node|'s cluster id to |cluster_id|
  void cluster_node(const size_t& node, const int& cluster_id);

 private:
  bool immutable_ = false;
  size_t num_nodes_;
  size_t num_threads_;
  std::shared_ptr<spdlog::logger> logger_ = nullptr;
#if defined(BIT_ADJ)
  std::vector<std::vector<uint64_t>> temp_adj_;
#else
  std::vector<std::vector<size_t>> temp_adj_;
#endif

  void constexpr assert_mutable_() const {
    if (immutable_) {
      throw std::runtime_error("Graph is immutable!");
    }
  }
  void constexpr assert_immutable_() const {
    if (!immutable_) {
      throw std::runtime_error("finalize is not called on graph!");
    }
  }
  void set_logger_() {
    logger_ = spdlog::get("console");
    if (logger_ == nullptr) {
      throw std::runtime_error("logger not created!");
    }
  }
};
}  // namespace GDBSCAN

#endif  // GDBSCAN_INCLUDE_GRAPH_H_
