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
#if defined(BIT_ADJ)
  explicit Graph(size_t num_nodes)
      : Va(num_nodes * 2, 0),
        // -1 as unvisited/un-clustered.
        cluster_ids(num_nodes, -1),
        membership(num_nodes, membership::Noise),
        num_nodes_(num_nodes) {
    set_logger_();
    // uint64_t is 64 bits; ceiling division.
    size_t num_uint64 = num_nodes_ / 64u + (num_nodes_ % 64u != 0);
    temp_adj_.resize(num_nodes_, std::vector<uint64_t>(num_uint64, 0u));
  }
#else
  explicit Graph(size_t num_nodes)
      : Va(num_nodes * 2, 0),
        // -1 as unvisited/un-clustered.
        cluster_ids(num_nodes, -1),
        membership(num_nodes, membership::Noise),
        temp_adj_(num_nodes, std::vector<size_t>()),
        num_nodes_(num_nodes) {
    set_logger_();
  }
#endif

#if defined(BIT_ADJ)
  void insert_edge(const size_t& u, const size_t& idx, const uint64_t& mask) {
    assert_mutable_();
    if (u >= num_nodes_ || idx >= temp_adj_[u].size()) {
      std::ostringstream oss;
      oss << "u=" << u << " or idx=" << idx << " is out of bound!";
      throw std::runtime_error(oss.str());
    }
    temp_adj_[u][idx] |= mask;
  }
#else
  void insert_edge(size_t u, size_t v) {
    assert_mutable_();
    if (u >= num_nodes_ || v >= num_nodes_) {
      std::ostringstream oss;
      oss << "u=" << u << " or v=" << v << " is out of bound!";
      throw std::runtime_error(oss.str());
    }
    logger_->debug("push {} as a neighbour of {}", v, u);
    temp_adj_[u].push_back(v);
  }
#endif

  void cluster_node(size_t node, int cluster_id) {
    assert_immutable_();
    if (node >= num_nodes_) {
      std::ostringstream oss;
      oss << node << " is out of bound!";
      throw std::runtime_error(oss.str());
    }
    cluster_ids[node] = cluster_id;
  }

#if defined(BIT_ADJ)
  void finalize() {
    logger_->info("finalize - BIT_ADJ");
    assert_mutable_();

    using namespace std::chrono;
    high_resolution_clock::time_point t0 = high_resolution_clock::now();

    for (size_t node = 0; node < num_nodes_; ++node) {
      // position in Ea
      Va[node * 2] = node == 0 ? 0 : (Va[node * 2 - 1] + Va[node * 2 - 2]);
      for (const uint64_t& val : temp_adj_[node]) {
        // number of neighbours
        Va[node * 2 + 1] += __builtin_popcountll(val);
      }
    }

    auto t1 = high_resolution_clock::now();
    auto d1 = duration_cast<duration<double> >(t1 - t0);
    logger_->info("\tconstructing Va takes {} seconds", d1.count());

    Ea.resize(Va[Va.size() - 1] + Va[Va.size() - 2], 0llu);
    // return if the graph has no edges.
    if (Ea.size() == 0u) {
      immutable_ = true;
      temp_adj_.clear();
      return;
    }

    auto it = std::begin(Ea);
    for (const auto& nbs : temp_adj_) {
      for (size_t i = 0; i < nbs.size(); ++i) {
        uint64_t val = nbs[i];
        logger_->debug("val is {}", val);
        // as fast as the branchless loop at worse case (complete graph), but
        // faster on average (skip all the 0 bits).
        while (val) {
          uint8_t k = __builtin_ffsll(val) - 1;
          *it = 64 * i + k;
          logger_->debug("k={}, *it={}", k, *it);
          ++it;
          val &= (val - 1);
        }
      }
    }
    assert(it == std::end(Ea) && "Humm it should be at the end of Ea");

    auto t2 = high_resolution_clock::now();
    auto d2 = duration_cast<duration<double> >(t2 - t1);
    logger_->info("\tconstructing Ea takes {} seconds", d2.count());

    immutable_ = true;
    temp_adj_.clear();
  }
#else   // BIT_ADJ
  void finalize() {
    logger_->info("finalize - DEFAULT");
    assert_mutable_();

    using namespace std::chrono;
    high_resolution_clock::time_point t0 = high_resolution_clock::now();

    size_t node = 0;
    for (const auto &nbs : temp_adj_) {
      // pos in Ea
      Va[node * 2] = node == 0 ? 0 : (Va[node * 2 - 1] + Va[node * 2 - 2]);
      // number of neighbours
      Va[node * 2 + 1] = nbs.size();
      ++node;
    }

    auto t1 = high_resolution_clock::now();
    auto d1 = duration_cast<duration<double> >(t1 - t0);
    logger_->info("\tconstructing Va takes {} seconds", d1.count());

    Ea.reserve(Va[Va.size() - 1] + Va[Va.size() - 2]);
    for (const auto &nbs : temp_adj_) {
      Ea.insert(Ea.end(), nbs.cbegin(), nbs.cend());
    }

    auto t2 = high_resolution_clock::now();
    auto d2 = duration_cast<duration<double> >(t2 - t1);
    logger_->info("\tconstructing Ea takes {} seconds", d2.count());

    immutable_ = true;
    temp_adj_.clear();
  }
#endif  // BIT_ADJ

 private:
#if defined(BIT_ADJ)
  std::vector<std::vector<uint64_t> > temp_adj_;
#else
  std::vector<std::vector<size_t> > temp_adj_;
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
  bool immutable_ = false;
  size_t num_nodes_;
  std::shared_ptr<spdlog::logger> logger_ = nullptr;
};
}  // namespace GDBSCAN

#endif  // GDBSCAN_INCLUDE_GRAPH_H_
