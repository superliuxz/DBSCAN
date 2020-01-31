//
// Created by William Liu on 2020-01-23.
//

#ifndef GDBSCAN_INCLUDE_GRAPH_H_
#define GDBSCAN_INCLUDE_GRAPH_H_

#include <vector>

#include "Membership.h"

namespace GDBSCAN {

class Graph {
 public:
  std::vector<size_t> Va;
  std::vector<size_t> Ea;
  std::vector<int> cluster_ids;
  std::vector<membership::Membership> membership;

  explicit Graph(size_t num_nodes) :
      Va(num_nodes * 2, 0),
      // -1 as unvisited/un-clustered.
      cluster_ids(num_nodes, -1),
#ifndef OPTM_2
      membership(num_nodes, membership::Border),
      Ea_(num_nodes, std::vector<size_t>()) {}
#else
  membership(num_nodes, membership::Border) {
  Ea_.reserve(num_nodes);
  for (size_t i = 0; i < num_nodes; ++i) {
    std::vector<size_t> n;
    n.reserve(num_nodes - 1);
    Ea_.emplace_back(n);
  }
}
#endif

  void insert_edge(size_t u, size_t v) {
    assert_mutable();
    if (u >= Ea_.size() || v >= Ea_.size()) {
      std::ostringstream oss;
      oss << u << "-" << v << " is out of bound!";
      throw std::runtime_error(oss.str());
    }
    Ea_[u].push_back(v);
    Ea_[v].push_back(u);
  }

  void cluster_node(size_t node, int cluster_id) {
    assert_immutable();
    if (node >= cluster_ids.size()) {
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
          static_cast<size_t>(
              curr_node ==
                  0 ? 0 : (Va[curr_node * 2 - 1] + Va[curr_node * 2 - 2])
          );
#ifndef OPTM_1
      for (const auto &nb: nbs) {
        Ea.push_back(nb);
      }
#endif
      ++curr_node;
    }
#ifdef OPTM_1
    size_t Ea_size = Va[Va.size() - 1] + Va[Va.size() - 2];
    Ea.reserve(Ea_size);

    for (const auto &nbs: Ea_) {
      for (const auto &nb: nbs) {
        Ea.push_back(nb);
      }
    }
#endif

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
