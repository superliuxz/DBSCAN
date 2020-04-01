//
// Created by William Liu on 2020-01-23.
//

#ifndef DBSCAN_INCLUDE_GRAPH_H_
#define DBSCAN_INCLUDE_GRAPH_H_

#include <spdlog/spdlog.h>

#include <vector>

#include "membership.h"
#include "utils.h"

namespace DBSCAN {

class Graph {
 public:
  std::vector<size_t> Va;
  std::vector<size_t, DBSCAN::utils::NonConstructAllocator<size_t>> Ea;
  std::vector<int> cluster_ids;
  std::vector<membership> memberships;
  // ctor
  explicit Graph(const size_t&, const size_t&);
  // insert edge
#if defined(BIT_ADJ)
  void insert_edge(const size_t&, const size_t&, const uint64_t&);
#else
  void start_insert(const size_t& u) { temp_adj_[u].reserve(num_vtx_); }
  void insert_edge(const size_t&, const size_t&);
  void finish_insert(const size_t& u) { temp_adj_[u].shrink_to_fit(); }
#endif
  // construct Va and Ea.
  void finalize();
  // set |vertex|'s cluster id to |cluster_id|
  void cluster_vertex(const size_t& vertex, const int& cluster_id);

 private:
  bool immutable_ = false;
  size_t num_vtx_;
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
}  // namespace DBSCAN

#endif  // DBSCAN_INCLUDE_GRAPH_H_
