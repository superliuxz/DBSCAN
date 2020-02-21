//
// Created by William Liu on 2020-01-23.
//

#ifndef GDBSCAN_INCLUDE_GRAPH_H_
#define GDBSCAN_INCLUDE_GRAPH_H_

#include <spdlog/spdlog.h>

#include <bitset>
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

#if defined(FLAT_ADJ)
  explicit Graph(size_t num_nodes)
      : Va(num_nodes * 2, 0),
        // -1 as unvisited/un-clustered.
        cluster_ids(num_nodes, -1),
        membership(num_nodes, membership::Noise),
        num_nodes_(num_nodes),
        temp_adj_list_(num_nodes * num_nodes, 0) {
    set_logger_();
    for (size_t i = 0; i < temp_adj_list_.size(); i += num_nodes) {
      temp_adj_list_[i] = i + 1;
    }
  }
#elif defined(BOOL_ADJ)
  explicit Graph(size_t num_nodes)
      : Va(num_nodes * 2, 0),
        // -1 as unvisited/un-clustered.
        cluster_ids(num_nodes, -1),
        membership(num_nodes, membership::Noise),
        num_nodes_(num_nodes) {
    set_logger_();
    const size_t r = num_nodes_ % 8;
    const size_t padding = r ? 8 - r : 0;
    temp_adj_list_.resize(num_nodes_,
                          std::vector<bool>(num_nodes_ + padding, false));
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
#elif defined(BITSET_ADJ)
  explicit Graph(size_t num_nodes)
      : Va(num_nodes * 2, 0),
        // -1 as unvisited/un-clustered.
        cluster_ids(num_nodes, -1),
        membership(num_nodes, membership::Noise),
        num_nodes_(num_nodes) {
    set_logger_();
    size_t num_bitset = num_nodes_ / 64u + (num_nodes_ % 64u != 0);
    temp_adj_list_.resize(
        num_nodes_, std::vector<std::bitset<64> >(num_bitset, 0x00000000));
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

#if defined(FLAT_ADJ)  // vector of size_t (node numbers)
    logger_->debug("push {} as a neighbour of {}", v, u);
    temp_adj_list_[temp_adj_list_[u * num_nodes_]++] = v;

#if defined(TRIA_ENUM)
    logger_->debug("push {} as a neighbour of {}", u, v);
    temp_adj_list_[temp_adj_list_[v * num_nodes_]++] = u;
#endif  // TRIA_ENUM

#elif defined(BOOL_ADJ)
    temp_adj_list_[u][v] = true;

#if defined(TRIA_ENUM)
    temp_adj_list_[v][u] = true;
#endif

#elif defined(BIT_ADJ)  // vector of vector of uint64
    size_t vidx = v / 64u;
    uint8_t voffset = v % 64u;  // 8bits -> 0-255, which is enough
    uint64_t vmask = 1llu << voffset;
    temp_adj_list_[u][vidx] |= vmask;
    logger_->debug("insert {} to neighbours of {}", v, u);
    logger_->debug("vidx {}, voffset {} vmask {:b}", vidx, voffset, vmask);

#if defined(TRIA_ENUM)
    size_t uidx = u / 64u;
    uint8_t uoffset = u % 64u;
    uint64_t umask = 1llu << uoffset;
    temp_adj_list_[v][uidx] |= umask;
    logger_->debug("insert {} to neighbours of {}", u, v);
    logger_->debug("uidx {}, uoffset {} umask {:b}", uidx, uoffset, umask);
#endif  // TRIA_ENUM

#elif defined(BITSET_ADJ)
    size_t vidx = v / 64u;
    uint8_t voffset = v % 64u;  // 8bits -> 0-255, which is enough
    temp_adj_list_[u][vidx].set(voffset);
    logger_->debug("insert {} to neighbours of {}", v, u);
    logger_->debug("vidx {}, voffset {}", vidx, voffset);

#if defined(TRIA_ENUM)
    size_t uidx = u / 64u;
    uint8_t uoffset = u % 64u;
    temp_adj_list_[v][uidx].set(uoffset);
    logger_->debug("insert {} to neighbours of {}", u, v);
    logger_->debug("uidx {}, uoffset {}", uidx, uoffset);
#endif  // TRIA_ENUM

#else  // vector of vector of size_t
    logger_->debug("push {} as a neighbour of {}", v, u);
    temp_adj_list_[u].push_back(v);

#if defined(TRIA_ENUM)
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

#if defined(FLAT_ADJ)
  void finalize() {
    logger_->info("finalize - FLAT_ADJ");
    assert_mutable_();

    size_t node = 0;
    size_t Va_pos = 0;
    // construct Va
    for (size_t i = 0; i < temp_adj_list_.capacity(); i += num_nodes_, ++node) {
      size_t num_neighbours = temp_adj_list_[i] - i - 1;
      Va[node * 2] = num_neighbours;
      Va[node * 2 + 1] = Va_pos;
      Va_pos += num_neighbours;
    }

    // construct Ea
    Ea.reserve(Va[Va.size() - 1] + Va[Va.size() - 2]);
    node = 0;
    for (size_t i = 0; i < temp_adj_list_.capacity(); i += num_nodes_, ++node) {
      size_t num_neighbours = Va[node * 2];
      for (size_t j = 0; j < num_neighbours; ++j) {
        Ea.push_back(temp_adj_list_[i + 1 + j]);
      }
    }

    immutable_ = true;
    temp_adj_list_.clear();
  }
#elif defined(BOOL_ADJ)
  void finalize() {
    logger_->info("finalize - BOOL_ADJ");
    assert_mutable_();
    for (size_t node = 0; node < num_nodes_; ++node) {
      Va[node * 2] = GDBSCAN::helper::popcount_bool(temp_adj_list_[node]);
      Va[node * 2 + 1] = node == 0 ? 0 : (Va[node * 2 - 1] + Va[node * 2 - 2]);
    }
    Ea.reserve(Va[Va.size() - 1] + Va[Va.size() - 2]);
    for (const auto& nbs : temp_adj_list_) {
      const std::vector<size_t> temp{GDBSCAN::helper::true_pos(nbs)};
      Ea.insert(Ea.end(), temp.cbegin(), temp.cend());
    }
    immutable_ = true;
    temp_adj_list_.clear();
  }
#elif defined(BIT_ADJ)
  void finalize() {
    logger_->info("finalize - BIT_ADJ");
    assert_mutable_();
    for (size_t node = 0; node < num_nodes_; ++node) {
      for (const uint64_t& val : temp_adj_list_[node]) {
        Va[node * 2] += GDBSCAN::helper::popcount64(val);
      }
      Va[node * 2 + 1] = node == 0 ? 0 : (Va[node * 2 - 1] + Va[node * 2 - 2]);
    }
    Ea.reserve(Va[Va.size() - 1] + Va[Va.size() - 2]);
    for (const auto& nbs : temp_adj_list_) {
      for (size_t i = 0; i < nbs.size(); ++i) {
        const std::vector<size_t> temp{GDBSCAN::helper::bit_pos(nbs[i], i)};
        Ea.insert(Ea.end(), temp.cbegin(), temp.cend());
      }
    }
    immutable_ = true;
    temp_adj_list_.clear();
  }
#elif defined(BITSET_ADJ)
  void finalize() {
    logger_->info("finalize - BITSET_ADJ");
    assert_mutable_();
    for (size_t node = 0; node < num_nodes_; ++node) {
      for (const auto& val : temp_adj_list_[node]) {
        Va[node * 2] += val.count();
      }
      Va[node * 2 + 1] = node == 0 ? 0 : (Va[node * 2 - 1] + Va[node * 2 - 2]);
    }
    Ea.reserve(Va[Va.size() - 1] + Va[Va.size() - 2]);
    for (const auto& nbs : temp_adj_list_) {
      for (size_t i = 0; i < nbs.size(); ++i) {
        const std::vector<size_t> temp{
            GDBSCAN::helper::bit_pos(nbs[i].to_ullong(), i)};
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

    size_t node = 0;
    for (const auto &nbs : temp_adj_list_) {
      // number of neighbours
      Va[node * 2] = nbs.size();
      // pos in Ea
      Va[node * 2 + 1] = node == 0 ? 0 : (Va[node * 2 - 1] + Va[node * 2 - 2]);

#if !defined(OPTM_1)
      Ea.insert(Ea.end(), nbs.cbegin(), nbs.cend());
#endif  // OPTM_1
      ++node;
    }
#if defined(OPTM_1)
    logger_->info("OPTM_1: using separate loop to construct Ea");
    Ea.reserve(Va[Va.size() - 1] + Va[Va.size() - 2]);

    for (const auto &nbs : temp_adj_list_) {
      Ea.insert(Ea.end(), nbs.cbegin(), nbs.cend());
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
#if defined(FLAT_ADJ)
  std::vector<size_t> temp_adj_list_;
#elif defined(BOOL_ADJ)
  std::vector<std::vector<bool> > temp_adj_list_;
#elif defined(BIT_ADJ)
  std::vector<std::vector<uint64_t> > temp_adj_list_;
#elif defined(BITSET_ADJ)
  std::vector<std::vector<std::bitset<64> > > temp_adj_list_;
#else
  std::vector<std::vector<size_t> > temp_adj_list_;
#endif
};
}  // namespace GDBSCAN

#endif  // GDBSCAN_INCLUDE_GRAPH_H_
