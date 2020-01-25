//
// Created by William Liu on 2020-01-23.
//

#ifndef GDBSCAN_INCLUDE_GRAPH_H_
#define GDBSCAN_INCLUDE_GRAPH_H_

#include <vector>

#include "Membership.h"

namespace GDBSCAN {

// use uint8/16/32/64.
class Graph {
 public:
  std::vector<size_t> Va;
  std::vector<size_t> Ea;
  std::vector<int> cluster_ids;
  std::vector<membership::Membership> membership;

  explicit Graph(size_t num_nodes) :
      Va(num_nodes * 2, 0),
      Ea_(num_nodes, std::vector<size_t>()),
      membership(num_nodes, membership::Border),
      cluster_ids(num_nodes, 0) {
  }

  void insert_edge(size_t u, size_t v) {
    assert_mutable();
    if (u < 0 || u >= Ea_.size() || v < 0 || v >= Ea_.size()) {
      std::ostringstream oss;
      oss << u << "-" << v << " is out of bound!";
      throw std::runtime_error(oss.str());
    }
    Ea_[u].push_back(v);
    Ea_[v].push_back(u);
  }

  void cluster_node(size_t node, int cluster_id) {
    assert_immutable();
    if (node < 0 || node >= cluster_ids.size()) {
      std::ostringstream oss;
      oss << node << " is out of bound!";
      throw std::runtime_error(oss.str());
    }
    cluster_ids[node] = cluster_id;
  }

  void finalize() {
    assert_mutable();

    size_t curr_node = 0;
    for (const auto &nbs: Ea_) {
      // number of neighbours
      Va[curr_node * 2] = static_cast<size_t>(nbs.size());
      // pos in Ea
      Va[curr_node * 2 + 1] =
          static_cast<size_t>(curr_node == 0 ? 0 : Ea.size());
      for (const auto &nb: nbs) {
        Ea.emplace_back(nb);
      }
      ++curr_node;
    }

    immutable_ = true;
    Ea_.clear();
  }

 private:
  void constexpr assert_mutable() const {
    if (immutable_) {
      throw std::runtime_error("Graph is immutable!");
    }
  }
  void constexpr assert_immutable() const {
    if (!immutable_) {
      throw std::runtime_error("finalize is not called on graph!");
    }
  }
  bool immutable_ = false;
  std::vector<std::vector<size_t>> Ea_;
};
} // namespace GDBSCAN

#endif //GDBSCAN_INCLUDE_GRAPH_H_