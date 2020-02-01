//
// Created by William Liu on 2020-01-23.
//

#ifndef GDBSCAN_INCLUDE_GRAPH_H_
#define GDBSCAN_INCLUDE_GRAPH_H_

#include <vector>
#include <spdlog/spdlog.h>

#include "Membership.h"
#include "Helper.h"

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
      membership(num_nodes, membership::Border),
      num_nodes_(num_nodes),
#ifndef OPTM_2
      temp_adj_list_(num_nodes, std::vector<size_t>()) {
#else
      temp_adj_list_(num_nodes * num_nodes, 0) {
        for (size_t i = 0; i < temp_adj_list_.capacity(); i += num_nodes) {
          temp_adj_list_[i] = i + 1;
        }
#endif
    logger_ = spdlog::get("console");
    if (logger_ == nullptr) {
      throw std::runtime_error("logger not created!");
    }
  }

  void insert_edge(size_t u, size_t v) {
    assert_mutable();

    if (u >= num_nodes_ || v >= num_nodes_) {
      std::ostringstream oss;
      oss << u << "-" << v << " is out of bound!";
      throw std::runtime_error(oss.str());
    }
#ifndef OPTM_2
    temp_adj_list_[u].push_back(v);
    temp_adj_list_[v].push_back(u);
#else
    size_t u_start = u * num_nodes_;
    size_t v_start = v * num_nodes_;
    temp_adj_list_[temp_adj_list_[u_start]++] = v;
    temp_adj_list_[temp_adj_list_[v_start]++] = u;
#endif // OPTM_2
  }

  void cluster_node(size_t node, int cluster_id) {
    assert_immutable();
    if (node >= num_nodes_) {
      std::ostringstream oss;
      oss << node << " is out of bound!";
      throw std::runtime_error(oss.str());
    }
    cluster_ids[node] = cluster_id;
  }

#ifndef OPTM_2
  void finalize() {
    assert_mutable();

    size_t curr_node = 0;
    for (const auto &nbs: temp_adj_list_) {
      // number of neighbours
      Va[curr_node * 2] = nbs.size();
      // pos in Ea
      Va[curr_node * 2 + 1] =
          curr_node == 0 ? 0 : (Va[curr_node * 2 - 1] + Va[curr_node * 2 - 2]);

#ifndef OPTM_1
      for (const auto &nb: nbs) {
        Ea.push_back(nb);
      }
#endif // OPTM_1
      ++curr_node;
    }
#ifdef OPTM_1
    size_t Ea_size = Va[Va.size() - 1] + Va[Va.size() - 2];
    Ea.reserve(Ea_size);

    for (const auto &nbs: temp_adj_list_) {
      for (const auto &nb: nbs) {
        Ea.push_back(nb);
      }
    }
#endif // OPTM_1

    immutable_ = true;
    temp_adj_list_.clear();
  }
#else
  void finalize() {
    assert_mutable();

    size_t curr_node = 0;
    size_t Va_pos = 0;
    // construct Va
    for (size_t i = 0; i < temp_adj_list_.capacity();
         i += num_nodes_, ++curr_node) {
      size_t num_neighbours = temp_adj_list_[i] - i - 1;
      Va[curr_node * 2] = num_neighbours;
      Va[curr_node * 2 + 1] = Va_pos;
      Va_pos += num_neighbours;
    }

    // construct Ea
    Ea.reserve(Va[Va.size() - 1] + Va[Va.size() - 2]);
    curr_node = 0;
    for (size_t i = 0; i < temp_adj_list_.capacity();
         i += num_nodes_, ++curr_node) {
      size_t num_neighbours = Va[curr_node * 2];
      for (size_t j = 0; j < num_neighbours; ++j) {
        Ea.push_back(temp_adj_list_[i + 1 + j]);
      }
    }

    immutable_ = true;
    temp_adj_list_.clear();
  }
#endif // OPTM_2

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
  size_t num_nodes_;
  std::shared_ptr<spdlog::logger> logger_ = nullptr;
#ifndef OPTM_2
  std::vector<std::vector<size_t>> temp_adj_list_;
#else
  std::vector<size_t> temp_adj_list_;
#endif
};
} // namespace GDBSCAN

#endif //GDBSCAN_INCLUDE_GRAPH_H_
