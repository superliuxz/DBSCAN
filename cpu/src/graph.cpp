//
// Created by William Liu on 2020-03-18.
//

#include "graph.h"

#include <spdlog/spdlog.h>

#include <execution>
#include <vector>

#include "DBSCAN/utils.h"

// ctor
#if defined(BIT_ADJ)
DBSCAN::Graph::Graph(const uint64_t num_vtx, const uint8_t num_threads)
    : num_nbs(num_vtx, 0),
      start_pos(num_vtx, 0),
      // -1 as unvisited/un-clustered.
      num_vtx_(num_vtx),
      num_threads_(num_threads) {
  SetLogger_();
  uint64_t num_uint64 = std::ceil(num_vtx_ / 64.0f);
  temp_adj_.resize(num_vtx_, std::vector<uint64_t>(num_uint64, 0u));
}
#else
DBSCAN::Graph::Graph(const uint64_t num_vtx, const uint8_t num_threads)
    : num_nbs(num_vtx, 0),
      start_pos(num_vtx, 0),
      num_vtx_(num_vtx),
      num_threads_(num_threads),
      temp_adj_(num_vtx, std::vector<uint64_t>()) {
  SetLogger_();
}
#endif

// insert edge
#if defined(BIT_ADJ)
void DBSCAN::Graph::InsertEdge(const uint64_t u, const uint64_t idx,
                               const uint64_t mask) {
  AssertMutable_();
  if (u >= num_vtx_ || idx >= temp_adj_[u].size()) {
    std::ostringstream oss;
    oss << "u=" << u << " or idx=" << idx << " is out of bound!";
    throw std::runtime_error(oss.str());
  }
  // logger_->trace("push {} as a neighbour of {}", idx * 64 + 63 -
  // __builtin_clzll(mask), u);
  temp_adj_[u][idx] |= mask;
}
#else
void DBSCAN::Graph::InsertEdge(const uint64_t u, const uint64_t v) {
  AssertMutable_();
  if (u >= num_vtx_ || v >= num_vtx_) {
    std::ostringstream oss;
    oss << "u=" << u << " or v=" << v << " is out of bound!";
    throw std::runtime_error(oss.str());
  }
  // logger_->trace("push {} as a neighbour of {}", v, u);
  temp_adj_[u].push_back(v);
}
#endif

#if defined(BIT_ADJ)
void DBSCAN::Graph::Finalize() {
  logger_->info("finalize - BIT_ADJ");
  AssertMutable_();

  using namespace std::chrono;
  auto t0 = high_resolution_clock::now();

  // Used for exclusive_scan to construct start_pos vector. "binary_op(init,
  // *first), binary_op(init, init), and binary_op(*first, *first) must be
  // convertible to T" -
  // https://en.cppreference.com/w/cpp/algorithm/exclusive_scan
  struct bin_op {
    uint64_t operator()(uint64_t x, uint64_t y) { return x + y; }
    uint64_t operator()(uint64_t x, const std::vector<uint64_t>& nbs) {
      uint64_t total = 0;
      for (const uint64_t val : nbs) total += __builtin_popcountll(val);
      return total + x;
    }
    uint64_t operator()(const std::vector<uint64_t>& a,
                        const std::vector<uint64_t>& b) {
      uint64_t total = 0;
      for (const uint64_t val : a) total += __builtin_popcountll(val);
      for (const uint64_t val : b) total += __builtin_popcountll(val);
      return total;
    }
  };
  std::exclusive_scan(std::execution::par_unseq, temp_adj_.cbegin(),
                      temp_adj_.cend(), start_pos.begin(), 0, bin_op{});
  std::transform(std::execution::par_unseq, temp_adj_.cbegin(),
                 temp_adj_.cend(), num_nbs.begin(),
                 [](const std::vector<uint64_t>& nbs) {
                   uint64_t total = 0;
                   for (const uint64_t val : nbs)
                     total += __builtin_popcountll(val);
                   return total;
                 });

  auto t1 = high_resolution_clock::now();
  auto d1 = duration_cast<duration<double>>(t1 - t0);
  logger_->info("\tconstructing num_nbs takes {} seconds", d1.count());

  const uint64_t sz = num_nbs.back() + start_pos.back();
  // return if the graph has no edges.
  if (sz == 0u) {
    temp_adj_.clear();
    temp_adj_.shrink_to_fit();
    immutable_ = true;
    return;
  }

  neighbours.resize(sz);

  auto t2 = high_resolution_clock::now();
  auto d2 = duration_cast<duration<double>>(t2 - t1);
  logger_->info("\tInit neighbours takes {} seconds", d2.count());

  std::vector<std::thread> threads(num_threads_);
  for (uint8_t tid = 0; tid < num_threads_; ++tid) {
    // logger_->debug("\tspawning thread {}", tid);
    threads[tid] = std::thread(
        [this](const uint8_t tid) {
          auto p_t0 = high_resolution_clock::now();
          for (uint64_t u = tid; u < num_vtx_; u += num_threads_) {
            const std::vector<uint64_t>& nbs = temp_adj_[u];
            auto it = std::next(neighbours.begin(), start_pos[u]);
            for (uint64_t i = 0; i < nbs.size(); ++i) {
              uint64_t val = nbs[i];
              while (val) {
                uint8_t k = __builtin_ffsll(val) - 1;
                *it = 64 * i + k;
                // logger_->trace("k={}, *it={}", k, *it);
                ++it;
                val &= (val - 1);
              }
            }
            // assert(static_cast<uint64_t>(std::distance(
            //        neighbours.begin(), it)) == num_nbs[u] + start_pos[u] &&
            //        "iterator steps != num_nbs[u]+start_pos[u]");
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
  logger_->info("\tCalc neighbours takes {} seconds", d3.count());

  temp_adj_.clear();
  temp_adj_.shrink_to_fit();
  immutable_ = true;
}
#else
void DBSCAN::Graph::Finalize() {
  logger_->info("Finalize - DEFAULT");
  AssertMutable_();

  using namespace std::chrono;
  auto t0 = high_resolution_clock::now();

  // Used for exclusive_scan to construct start_pos vector. "binary_op(init,
  // *first), binary_op(init, init), and binary_op(*first, *first) must be
  // convertible to T" -
  // https://en.cppreference.com/w/cpp/algorithm/exclusive_scan
  struct bin_op {
    uint64_t operator()(uint64_t x, uint64_t y) { return x + y; }
    uint64_t operator()(uint64_t x, const std::vector<uint64_t>& nbs) {
      return x + nbs.size();
    }
    uint64_t operator()(const std::vector<uint64_t>& a,
                        const std::vector<uint64_t>& b) {
      return a.size() + b.size();
    }
  };
  std::exclusive_scan(std::execution::par_unseq, temp_adj_.cbegin(),
                      temp_adj_.cend(), start_pos.begin(), 0, bin_op{});
  std::transform(std::execution::par_unseq, temp_adj_.cbegin(),
                 temp_adj_.cend(), num_nbs.begin(),
                 [](const std::vector<uint64_t>& nbs) { return nbs.size(); });

  auto t1 = high_resolution_clock::now();
  auto d1 = duration_cast<duration<double>>(t1 - t0);
  logger_->info("\tCalc num_nbs takes {} seconds", d1.count());

  const uint64_t sz = num_nbs.back() + start_pos.back();
  // return if the graph has no edges.
  if (sz == 0u) {
    temp_adj_.clear();
    temp_adj_.shrink_to_fit();
    immutable_ = true;
    return;
  }

  neighbours.resize(sz);

  auto t2 = high_resolution_clock::now();
  auto d2 = duration_cast<duration<double>>(t2 - t1);
  logger_->info("\tInit neighbours takes {} seconds", d2.count());

  std::vector<std::thread> threads(num_threads_);
  for (uint8_t tid = 0; tid < num_threads_; ++tid) {
    // logger_->debug("\tspawning thread {}", tid);
    threads[tid] = std::thread(
        [this](const uint8_t tid) {
          auto p_t0 = high_resolution_clock::now();
          for (uint64_t u = tid; u < num_vtx_; u += num_threads_) {
            const auto& nbs = temp_adj_[u];
            // logger_->trace("\twriting vtx {} with # nbs {}", u, nbs.size());
            assert(nbs.size() == num_nbs[u] && "nbs.size!=num_nbs[u]");
            std::copy(nbs.cbegin(), nbs.cend(),
                      neighbours.begin() + start_pos[u]);
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
  logger_->info("\tCalc neighbours takes {} seconds", d3.count());

  temp_adj_.clear();
  temp_adj_.shrink_to_fit();
  immutable_ = true;
}
#endif