//
// Created by William Liu on 2020-01-23.
//

#ifndef GDBSCAN_INCLUDE_GRAPH_H_
#define GDBSCAN_INCLUDE_GRAPH_H_

#include <spdlog/spdlog.h>

#include <vector>

#include "Helper.h"
#include "Membership.h"

namespace GDBSCAN {

class Graph {
 public:
  std::vector<size_t> Va;
  std::vector<size_t> Ea;
  std::vector<int> cluster_ids;
  std::vector<membership::Membership> membership;

#ifdef FLAT_ADJ
  explicit Graph(size_t num_nodes)
      : Va(num_nodes * 2, 0),
        // -1 as unvisited/un-clustered.
        cluster_ids(num_nodes, -1),
        membership(num_nodes, membership::Border),
        num_nodes_(num_nodes),
        temp_adj_list_(num_nodes * num_nodes, 0) {
    set_logger_();
    for (size_t i = 0; i < temp_adj_list_.size(); i += num_nodes) {
      temp_adj_list_[i] = i + 1;
    }
  }
#elif defined(BIT_ADJ)
  explicit Graph(size_t num_nodes)
      : Va(num_nodes * 2, 0),
        // -1 as unvisited/un-clustered.
        cluster_ids(num_nodes, -1),
        membership(num_nodes, membership::Noise),
        num_nodes_(num_nodes) {
    set_logger_();
    // uint64_t is 64 bits; ceiling division.
    size_t num_uint64 = num_nodes_ / 64u + (num_nodes_ % 64u != 0);
    temp_adj_list_.resize(num_nodes_, std::vector<uint64_t>(num_uint64, 0u));
  }

#else
  explicit Graph(size_t num_nodes)
      : Va(num_nodes * 2, 0),
        // -1 as unvisited/un-clustered.
        cluster_ids(num_nodes, -1),
        membership(num_nodes, membership::Noise),
        num_nodes_(num_nodes),
        temp_adj_list_(num_nodes, std::vector<size_t>()) {
    set_logger_();
  }
#endif

  void insert_edge(size_t u, size_t v) {
    assert_mutable_();

    if (u >= num_nodes_ || v >= num_nodes_) {
      std::ostringstream oss;
      oss << u << "-" << v << " is out of bound!";
      throw std::runtime_error(oss.str());
    }

#ifdef FLAT_ADJ  // vector of size_t (node numbers)
    logger_->debug("push {} as a neighbour of {}", v, u);
    temp_adj_list_[temp_adj_list_[u * num_nodes_]++] = v;

#ifdef TRIA_ENUM
    logger_->debug("push {} as a neighbour of {}", u, v);
    temp_adj_list_[temp_adj_list_[v * num_nodes_]++] = u;
#endif  // TRIA_ENUM

#elif defined(BIT_ADJ)  // vector of vector of uint64
    size_t vidx = v / 64u;
    uint8_t voffset = v % 64u;  // 8bits -> 0-255, which is enough
    uint64_t vmask = 1llu << voffset;
    temp_adj_list_[u][vidx] |= vmask;
    logger_->debug("insert {} to neighbours of {}", v, u);
    logger_->debug("vidx {}, voffset {} vmask {:b}", vidx, voffset, vmask);

#ifdef TRIA_ENUM
    size_t uidx = u / 64u;
    uint8_t uoffset = u % 64u;
    uint64_t umask = 1llu << uoffset;
    temp_adj_list_[v][uidx] |= umask;
    logger_->debug("insert {} to neighbours of {}", u, v);
    logger_->debug("uidx {}, uoffset {} umask {:b}", uidx, uoffset, umask);
#endif  // TRIA_ENUM

#else  // vector of vector of size_t
    logger_->debug("push {} as a neighbour of {}", v, u);
    temp_adj_list_[u].push_back(v);

#ifdef TRIA_ENUM
    logger_->debug("push {} as a neighbour of {}", u, v);
    temp_adj_list_[v].push_back(u);
#endif  // TRIA_ENUM

#endif  // FLAT_ADJ
  }

  void cluster_node(size_t node, int cluster_id) {
    assert_immutable_();
    if (node >= num_nodes_) {
      std::ostringstream oss;
      oss << node << " is out of bound!";
      throw std::runtime_error(oss.str());
    }
    cluster_ids[node] = cluster_id;
  }

#ifdef FLAT_ADJ
  void finalize() {
    logger_->info("finalize - FLAT_ADJ");
    assert_mutable_();

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
#elif defined(BIT_ADJ)
  void finalize() {
    logger_->info("finalize - BIT_ADJ");
    assert_mutable_();
    for (size_t curr_node = 0; curr_node < num_nodes_; ++curr_node) {
      for (const uint64_t& val : temp_adj_list_[curr_node]) {
        Va[curr_node * 2] += GDBSCAN::helper::popcount64(val);
      }
      Va[curr_node * 2 + 1] =
          curr_node == 0 ? 0 : (Va[curr_node * 2 - 1] + Va[curr_node * 2 - 2]);
    }
    Ea.reserve(Va[Va.size() - 1] + Va[Va.size() - 2]);
    for (size_t curr_node = 0; curr_node < num_nodes_; ++curr_node) {
      const std::vector<uint64_t>& vals = temp_adj_list_[curr_node];
      for (size_t i = 0; i < vals.size(); ++i) {
        const std::vector<size_t> temp{GDBSCAN::helper::bit_pos(vals[i], i)};
        Ea.insert(Ea.end(), temp.cbegin(), temp.cend());
      }
    }
    immutable_ = true;
    temp_adj_list_.clear();
  }
#else  // FLAT_ADJ
  void finalize() {
    logger_->info("finalize - DEFAULT");
    assert_mutable_();

    size_t curr_node = 0;
    for (const auto &nbs : temp_adj_list_) {
      // number of neighbours
      Va[curr_node * 2] = nbs.size();
      // pos in Ea
      Va[curr_node * 2 + 1] =
          curr_node == 0 ? 0 : (Va[curr_node * 2 - 1] + Va[curr_node * 2 - 2]);

#ifndef OPTM_1
      for (const auto &nb : nbs) {
        Ea.push_back(nb);
      }
#endif  // OPTM_1
      ++curr_node;
    }
#ifdef OPTM_1
    logger_->info("OPTM_1: using separate loop to construct Ea");
    size_t Ea_size = Va[Va.size() - 1] + Va[Va.size() - 2];
    Ea.reserve(Ea_size);

    for (const auto &nbs : temp_adj_list_) {
      for (const auto &nb : nbs) {
        Ea.push_back(nb);
      }
    }
#endif  // OPTM_1
    immutable_ = true;
    temp_adj_list_.clear();
  }
#endif  // FLAT_ADJ

 private:
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
  bool immutable_ = false;
  size_t num_nodes_;
  std::shared_ptr<spdlog::logger> logger_ = nullptr;
#ifdef FLAT_ADJ
  std::vector<size_t> temp_adj_list_;
#elif defined(BIT_ADJ)
  std::vector<std::vector<uint64_t> > temp_adj_list_;
#else
  std::vector<std::vector<size_t> > temp_adj_list_;
#endif
};
}  // namespace GDBSCAN

#endif  // GDBSCAN_INCLUDE_GRAPH_H_
