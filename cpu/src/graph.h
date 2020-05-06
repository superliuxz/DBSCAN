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
  std::vector<uint64_t> num_nbs;
  std::vector<uint64_t> start_pos;
  std::vector<uint64_t, DBSCAN::utils::NonConstructAllocator<uint64_t>>
      neighbours;
  // ctor
  explicit Graph(uint64_t, uint8_t);
  // insert edge
#if defined(BIT_ADJ)
  void InsertEdge(uint64_t, uint64_t, uint64_t);
#else
  void StartInsert(const uint64_t u) { temp_adj_[u].reserve(num_vtx_); }
  void InsertEdge(uint64_t, uint64_t);
  void FinishInsert(const uint64_t u) { temp_adj_[u].shrink_to_fit(); }
#endif
  // construct num_nbs and neighbours.
  void Finalize();

 private:
  bool immutable_ = false;
  uint64_t num_vtx_;
  uint8_t num_threads_;
  std::shared_ptr<spdlog::logger> logger_ = nullptr;
  std::vector<std::vector<uint64_t>> temp_adj_;

  void constexpr AssertMutable_() const {
    if (immutable_) {
      throw std::runtime_error("Graph is immutable!");
    }
  }
  void SetLogger_() {
    logger_ = spdlog::get("console");
    if (logger_ == nullptr) {
      throw std::runtime_error("logger not created!");
    }
  }
};
}  // namespace DBSCAN

#endif  // DBSCAN_INCLUDE_GRAPH_H_
