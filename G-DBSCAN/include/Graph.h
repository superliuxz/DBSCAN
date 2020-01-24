//
// Created by William Liu on 2020-01-23.
//

#ifndef GDBSCAN_INCLUDE_GRAPH_H_
#define GDBSCAN_INCLUDE_GRAPH_H_

#include <vector>

namespace GDBSCAN {

// use uint8/16/32/64.
template<class DataType>
class Graph {
 public:
  std::vector<DataType> Va;
  std::vector<DataType> Ea;
  std::vector<int> cluster_ids;

  explicit Graph(size_t num_nodes) {
    if (!std::is_same_v<DataType, uint8_t> &&
        !std::is_same_v<DataType, uint16_t> &&
        !std::is_same_v<DataType, uint32_t> &&
        !std::is_same_v<DataType, uint64_t>) {
      throw std::runtime_error("Please use unsigned int!");
    }
    Va.assign(num_nodes * 2, 0);
    Ea_.assign(num_nodes, std::vector<DataType>());
    cluster_ids.assign(num_nodes, 0);
  }

  void insert_edge(DataType u, DataType v) {
    assert_mutable();
    if (u < 0 || u >= Ea_.size() || v < 0 || v >= Ea_.size()) {
      std::ostringstream oss;
      oss << u << "-" << v << " is out of bound!" << std::endl;
      throw std::runtime_error(oss.str());
    }
    Ea_[u].push_back(v);
    Ea_[v].push_back(u);
  }

  void classify_node(DataType node, int cluster_id) {
    assert_immutable();
    if (node < 0 || node >= cluster_ids.size()) {
      std::ostringstream oss;
      oss << node << " is out of bound!" << std::endl;
      throw std::runtime_error(oss.str());
    }
    cluster_ids[node] = cluster_id;
  }

  void finalize() {
    assert_mutable();

    DataType curr_node = 0;
    for (const auto &nbs: Ea_) {
      // number of neighbours
      Va[curr_node * 2] = static_cast<DataType>(nbs.size());
      // pos in Ea
      Va[curr_node * 2 + 1] =
          static_cast<DataType>(curr_node == 0 ? 0 : Ea.size());
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
  std::vector<std::vector<DataType>> Ea_;
};
} // namespace GDBSCAN

#endif //GDBSCAN_INCLUDE_GRAPH_H_
