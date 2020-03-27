//
// Created by William Liu on 2020-03-18.
//

#include "graph.h"

#include <spdlog/spdlog.h>

#include <vector>
#include <cmath>

#include "membership.h"
#include "utils.h"

// ctor
#if defined(BIT_ADJ)
DBSCAN::Graph::Graph(const size_t& num_nodes, const size_t& num_threads)
    : Va(num_nodes * 2, 0),
      // -1 as unvisited/un-clustered.
      cluster_ids(num_nodes, -1),
      memberships(num_nodes, membership::Noise),
      num_nodes_(num_nodes),
      num_threads_(num_threads) {
  set_logger_();
  size_t num_uint64 = ceil(num_nodes_ / 64f);
  temp_adj_.resize(num_nodes_, std::vector<uint64_t>(num_uint64, 0u));
}
#else
DBSCAN::Graph::Graph(const size_t& num_nodes, const size_t& num_threads)
    : Va(num_nodes * 2, 0),
      // -1 as unvisited/un-clustered.
      cluster_ids(num_nodes, -1),
      memberships(num_nodes, membership::Noise),
      num_nodes_(num_nodes),
      num_threads_(num_threads),
      temp_adj_(num_nodes, std::vector<size_t>()) {
  set_logger_();
}
#endif

// insert edge
#if defined(BIT_ADJ)
void DBSCAN::Graph::insert_edge(const size_t& u, const size_t& idx,
                                const uint64_t& mask) {
  assert_mutable_();
  if (u >= num_nodes_ || idx >= temp_adj_[u].size()) {
    std::ostringstream oss;
    oss << "u=" << u << " or idx=" << idx << " is out of bound!";
    throw std::runtime_error(oss.str());
  }
  // logger_->trace("push {} as a neighbour of {}", idx * 64 + 63 -
  // __builtin_clzll(mask), u);
  temp_adj_[u][idx] |= mask;
}
#else
void DBSCAN::Graph::insert_edge(const size_t& u, const size_t& v) {
  assert_mutable_();
  if (u >= num_nodes_ || v >= num_nodes_) {
    std::ostringstream oss;
    oss << "u=" << u << " or v=" << v << " is out of bound!";
    throw std::runtime_error(oss.str());
  }
  //    logger_->trace("push {} as a neighbour of {}", v, u);
  temp_adj_[u].push_back(v);
}
#endif

void DBSCAN::Graph::cluster_node(const size_t& node, const int& cluster_id) {
  assert_immutable_();
  if (node >= num_nodes_) {
    std::ostringstream oss;
    oss << node << " is out of bound!";
    throw std::runtime_error(oss.str());
  }
  cluster_ids[node] = cluster_id;
}

#if defined(BIT_ADJ)
void DBSCAN::Graph::finalize() {
  logger_->info("finalize - BIT_ADJ");
  assert_mutable_();

  using namespace std::chrono;
  auto t0 = high_resolution_clock::now();

  // TODO: exclusive scan
  for (size_t node = 0; node < num_nodes_; ++node) {
    // position in Ea
    Va[node * 2] = node == 0 ? 0 : (Va[node * 2 - 1] + Va[node * 2 - 2]);
    for (const uint64_t& val : temp_adj_[node]) {
      // number of neighbours
      Va[node * 2 + 1] += __builtin_popcountll(val);
    }
  }

  auto t1 = high_resolution_clock::now();
  auto d1 = duration_cast<duration<double>>(t1 - t0);
  logger_->info("\tconstructing Va takes {} seconds", d1.count());

  const size_t sz = Va[Va.size() - 1] + Va[Va.size() - 2];
  // return if the graph has no edges.
  if (sz == 0u) {
    temp_adj_.clear();
    temp_adj_.shrink_to_fit();
    immutable_ = true;
    return;
  }

  Ea.resize(sz);

  auto t2 = high_resolution_clock::now();
  auto d2 = duration_cast<duration<double>>(t2 - t1);
  logger_->info("\tInit Ea takes {} seconds", d2.count());

  std::vector<std::thread> threads(num_threads_);
  const size_t chunk = ceil(num_nodes_ / static_cast<double>(num_threads_));
  for (size_t tid = 0; tid < num_threads_; ++tid) {
    // logger_->debug("\tspawning thread {}", tid);
    threads[tid] = std::thread(
        [this, &chunk](const size_t& tid) {
          auto p_t0 = high_resolution_clock::now();
          const size_t start = tid * chunk;
          const size_t end = std::min(start + chunk, num_nodes_);
          for (size_t u = start; u < end; ++u) {
            const std::vector<uint64_t>& nbs = temp_adj_[u];
            auto it = std::next(Ea.begin(), Va[2 * u]);
            for (size_t i = 0; i < nbs.size(); ++i) {
              uint64_t val = nbs[i];
              while (val) {
                uint8_t k = __builtin_ffsll(val) - 1;
                *it = 64 * i + k;
                // logger_->trace("k={}, *it={}", k, *it);
                ++it;
                val &= (val - 1);
              }
            }
            assert(static_cast<size_t>(std::distance(Ea.begin(), it)) ==
                       Va[2 * u] + Va[2 * u + 1] &&
                   "iterator steps != Va[2*u+1]");
          }
          auto p_t1 = high_resolution_clock::now();
          logger_->info("\t\tThread {} takes {} seconds", tid,
                        duration_cast<duration<double>>(p_t1 - p_t0).count());
        } /* lambda */,
        tid);
  }
  for (auto& tr : threads) tr.join();
  // logger_->debug("\tjoined all threads");

  auto t3 = high_resolution_clock::now();
  auto d3 = duration_cast<duration<double>>(t3 - t2);
  logger_->info("\tCalc Ea takes {} seconds", d3.count());

  temp_adj_.clear();
  temp_adj_.shrink_to_fit();
  immutable_ = true;
}
#else
void DBSCAN::Graph::finalize() {
  logger_->info("finalize - DEFAULT");
  assert_mutable_();

  using namespace std::chrono;
  auto t0 = high_resolution_clock::now();

  size_t node = 0;
  // TODO: paralleled exclusive scan
  for (const auto& nbs : temp_adj_) {
    // pos in Ea
    Va[node * 2] = node == 0 ? 0 : (Va[node * 2 - 1] + Va[node * 2 - 2]);
    // number of neighbours
    Va[node * 2 + 1] = nbs.size();
    ++node;
  }

  auto t1 = high_resolution_clock::now();
  auto d1 = duration_cast<duration<double>>(t1 - t0);
  logger_->info("\tCalc Va takes {} seconds", d1.count());

  const size_t sz = Va[Va.size() - 1] + Va[Va.size() - 2];
  // return if the graph has no edges.
  if (sz == 0u) {
    temp_adj_.clear();
    temp_adj_.shrink_to_fit();
    immutable_ = true;
    return;
  }

  Ea.resize(sz);

  auto t2 = high_resolution_clock::now();
  auto d2 = duration_cast<duration<double>>(t2 - t1);
  logger_->info("\tInit Ea takes {} seconds", d2.count());

  std::vector<std::thread> threads(num_threads_);
  const size_t chunk = ceil(num_nodes_ / static_cast<double>(num_threads_));
  for (size_t tid = 0; tid < num_threads_; ++tid) {
    // logger_->debug("\tspawning thread {}", tid);
    threads[tid] = std::thread(
        [this, &chunk](const size_t& tid) {
          auto p_t0 = high_resolution_clock::now();
          const size_t start = tid * chunk;
          const size_t end = std::min(start + chunk, num_nodes_);
          for (size_t u = start; u < end; ++u) {
            const auto& nbs = temp_adj_[u];
            // logger_->trace("\twriting vtx {} with # nbs {}", u,
            // nbs.size());
            assert(nbs.size() == Va[2 * u + 1] && "nbs.size!=Va[2*u+1]");
            std::copy(nbs.cbegin(), nbs.cend(), Ea.begin() + Va[2 * u]);
          }
          auto p_t1 = high_resolution_clock::now();
          logger_->info("\t\tThread {} takes {} seconds", tid,
                        duration_cast<duration<double>>(p_t1 - p_t0).count());
        }, /* lambda */
        tid /* args to lambda */);
  }
  for (auto& tr : threads) tr.join();
  // logger_->debug("\tjoined all threads");

  auto t3 = high_resolution_clock::now();
  auto d3 = duration_cast<duration<double>>(t3 - t2);
  logger_->info("\tCalc Ea takes {} seconds", d3.count());

  temp_adj_.clear();
  temp_adj_.shrink_to_fit();
  immutable_ = true;
}
#endif