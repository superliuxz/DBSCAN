//
// Created by William Liu on 2020-01-24.
//

#ifndef GDBSCAN_INCLUDE_SOLVER_H_
#define GDBSCAN_INCLUDE_SOLVER_H_

#include <fstream>
#include <memory>
#include <thread>

#include "dataset.h"
#include "graph.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace GDBSCAN {

template <class DataType>
class Solver {
 public:
  explicit Solver(const std::string& input, const uint& min_pts,
                  const float& radius, const uint8_t& num_threads)
      : min_pts_(min_pts),
        squared_radius_(radius * radius),
        num_threads_(num_threads) {
    logger_ = spdlog::get("console");
    if (logger_ == nullptr) {
      throw std::runtime_error("logger not created!");
    }

    using namespace std::chrono;
    high_resolution_clock::time_point start = high_resolution_clock::now();

    auto ifs = std::ifstream(input);
    ifs >> num_nodes_;
    dataset_ = std::make_unique<DataType>(num_nodes_);
    size_t n;
    float x, y;
    if (std::is_same_v<DataType, input_type::TwoDimPoints>) {
      while (ifs >> n >> x >> y) {
        dataset_->d1[n] = x;
        dataset_->d2[n] = y;
      }
    } else {
      throw std::runtime_error("Implement your own input_type!");
    }

    duration<double> time_spent =
        duration_cast<duration<double>>(high_resolution_clock::now() - start);
    logger_->info("reading vertices takes {} seconds", time_spent.count());
  }

#if defined(TESTING)
  const DataType& dataset_view() const { return *dataset_; }
#endif

  [[nodiscard]] const Graph& graph_view() const { return *graph_; }

  /*
   * For each two nodes, if the distance is <= |squared_radius_|, insert them
   * into the graph (|temp_adj_|). Part of Algorithm 1 (Andrade et al).
   */
  void insert_edges() {
    using namespace std::chrono;
    high_resolution_clock::time_point start = high_resolution_clock::now();

    if (dataset_ == nullptr) {
      throw std::runtime_error("Call prepare_dataset to generate the dataset!");
    }

    graph_ = std::make_unique<Graph>(num_nodes_, num_threads_);

    std::vector<std::thread> threads(num_threads_);
    const auto dist = input_type::TwoDimPoints::euclidean_distance_square;
    const size_t chunk = num_nodes_ / num_threads_ + 1;
#if defined(BIT_ADJ)
    logger_->info("insert_edges - BIT_ADJ");
    const size_t N = num_nodes_ / 64u + (num_nodes_ % 64u != 0);
    for (size_t tid = 0; tid < num_threads_; ++tid) {
      threads[tid] = std::thread(
          [this, &N, &dist, &chunk](const size_t& tid) {
            auto t0 = high_resolution_clock::now();
            const size_t start = tid * chunk;
            const size_t end = std::min(start + chunk, num_nodes_);
            for (size_t u = start; u < end; ++u) {
              const float &ux = dataset_->d1[u], uy = dataset_->d2[u];
              for (size_t outer = 0; outer < N; outer += 4) {
                for (size_t inner = 0; inner < 64; ++inner) {
                  const size_t v1 = outer * 64llu + inner;
                  const size_t v2 = v1 + 64;
                  const size_t v3 = v2 + 64;
                  const size_t v4 = v3 + 64;
                  const uint64_t msk = 1llu << inner;
                  if (u != v1 && v1 < num_nodes_ &&
                      dist(ux, uy, dataset_->d1[v1], dataset_->d2[v1]) <=
                          squared_radius_)
                    graph_->insert_edge(u, outer, msk);
                  if (u != v2 && v2 < num_nodes_ &&
                      dist(ux, uy, dataset_->d1[v2], dataset_->d2[v2]) <=
                          squared_radius_)
                    graph_->insert_edge(u, outer + 1, msk);
                  if (u != v3 && v3 < num_nodes_ &&
                      dist(ux, uy, dataset_->d1[v3], dataset_->d2[v3]) <=
                          squared_radius_)
                    graph_->insert_edge(u, outer + 2, msk);
                  if (u != v4 && v4 < num_nodes_ &&
                      dist(ux, uy, dataset_->d1[v4], dataset_->d2[v4]) <=
                          squared_radius_)
                    graph_->insert_edge(u, outer + 3, msk);
                }
              }
            }
            auto t1 = high_resolution_clock::now();
            logger_->info("\tThread {} takes {} seconds", tid,
                          duration_cast<duration<double>>(t1 - t0).count());
          }, /* lambda */
          tid /* args to lambda */);
    }
#else
    logger_->info("insert_edges - default");
    for (size_t tid = 0; tid < num_threads_; ++tid) {
      threads[tid] = std::thread(
          [this, &dist, &chunk](const size_t& tid) {
            auto t0 = high_resolution_clock::now();
            const size_t start = tid * chunk;
            const size_t end = std::min(start + chunk, num_nodes_);
            for (size_t u = start; u < end; ++u) {
              const float &ux = dataset_->d1[u], uy = dataset_->d2[u];
              for (size_t v = 0; v < num_nodes_; ++v) {
                if (u != v && dist(ux, uy, dataset_->d1[v], dataset_->d2[v]) <=
                                  squared_radius_) {
                  graph_->insert_edge(u, v);
                }
              }
            }
            auto t1 = high_resolution_clock::now();
            logger_->info("\tThread {} takes {} seconds", tid,
                          duration_cast<duration<double>>(t1 - t0).count());
          }, /* lambda */
          tid /* args to lambda */);
    }
#endif
    for (auto& tr : threads) tr.join();
    threads.clear();

    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double> time_spent = duration_cast<duration<double>>(end - start);
    logger_->info("insert_edges (Algorithm 1) takes {} seconds",
                  time_spent.count());
  }

  /*
   * Construct |Va| and |Ea| from |temp_adj|. Part of Algorithm 1
   * (Andrade et al).
   */
  void finalize_graph() const {
    using namespace std::chrono;
    high_resolution_clock::time_point start = high_resolution_clock::now();

    graph_->finalize();

    duration<double> time_spent =
        duration_cast<duration<double>>(high_resolution_clock::now() - start);
    logger_->info("finalize_graph takes {} seconds", time_spent.count());
  }

  /*
   * Classify nodes to Core or Noise; the Border nodes are classified in the BFS
   * stage (Algorithm 2 in Andrade et al).
   */
  void classify_nodes() const {
    using namespace std::chrono;
    high_resolution_clock::time_point start = high_resolution_clock::now();

    if (graph_ == nullptr) {
      throw std::runtime_error("Call insert_edges to generate the graph!");
    }

    for (size_t node = 0; node < num_nodes_; ++node) {
      logger_->trace("{} has {} neighbours within {}", node,
                     graph_->Va[node * 2 + 1], squared_radius_);
      logger_->trace("{} >= {}: {}", graph_->Va[node * 2], min_pts_,
                     graph_->Va[node * 2 + 1] >= min_pts_ ? "true" : "false");
      if (graph_->Va[node * 2 + 1] >= min_pts_) {
        logger_->debug("{} to Core", node);
        graph_->membership[node] = Core;
      } else {
        logger_->debug("{} to Noise", node);
        graph_->membership[node] = Noise;
      }
    }

    duration<double> time_spent =
        duration_cast<duration<double>>(high_resolution_clock::now() - start);
    logger_->info("classify_nodes takes {} seconds", time_spent.count());
  }

  /*
   * Algorithm 2 (BFS) in Andrade et al.
   */
  void identify_cluster() const {
    using namespace std::chrono;
    high_resolution_clock::time_point start = high_resolution_clock::now();

    int cluster = 0;
    for (size_t node = 0; node < num_nodes_; ++node) {
      if (graph_->cluster_ids[node] == -1 && graph_->membership[node] == Core) {
        graph_->cluster_ids[node] = cluster;
        logger_->debug("start bfs on node {} with cluster {}", node, cluster);
        bfs(node, cluster);
        ++cluster;
      }
    }

    duration<double> time_spent =
        duration_cast<duration<double>>(high_resolution_clock::now() - start);
    logger_->info("identify_cluster (Algorithm 2) takes {} seconds",
                  time_spent.count());
  }

 private:
  size_t num_nodes_{};
  size_t min_pts_{};
  float squared_radius_{};
  uint8_t num_threads_{};
  std::unique_ptr<DataType> dataset_ = nullptr;
  std::unique_ptr<Graph> graph_ = nullptr;
  std::shared_ptr<spdlog::logger> logger_ = nullptr;

  /*
   * BFS. Start from |node| and visit all the reachable neighbours. If a
   * neighbour is Noise, relabel it to Border.
   */
  void bfs(size_t start_node, int cluster) const {
    std::vector<size_t> curr_level{start_node};
    // each thread has its own partial frontier.
    std::vector<std::vector<size_t>> next_level(num_threads_,
                                                std::vector<size_t>());

    std::vector<std::thread> threads(num_threads_);
    size_t lvl_cnt = 0;
    while (!curr_level.empty()) {
      logger_->info("\tBFS level {}", lvl_cnt);
      for (size_t tid = 0u; tid < num_threads_; ++tid) {
        threads[tid] = std::thread(
            [this, &curr_level, &next_level, &cluster](const size_t& tid) {
              using namespace std::chrono;
              auto start = high_resolution_clock::now();
              for (size_t curr_node_idx = tid;
                   curr_node_idx < curr_level.size();
                   curr_node_idx += num_threads_) {
                size_t node = curr_level[curr_node_idx];
                logger_->trace("visiting node {}", node);
                // Relabel a reachable Noise node, but do not keep exploring.
                if (graph_->membership[node] == Noise) {
                  logger_->trace("\tnode {} is relabeled from Noise to Border",
                                 node);
                  graph_->membership[node] = Border;
                  continue;
                }
                size_t start_pos = graph_->Va[2 * node];
                size_t num_neighbours = graph_->Va[2 * node + 1];
                for (size_t i = 0; i < num_neighbours; ++i) {
                  size_t nb = graph_->Ea[start_pos + i];
                  if (graph_->cluster_ids[nb] == -1) {
                    // cluster the node
                    logger_->trace("\tnode {} is clustered tp {}", nb, cluster);
                    graph_->cluster_ids[nb] = cluster;
                    logger_->trace("\tneighbour {} of node {} is queued", nb,
                                   node);
                    next_level[tid].emplace_back(nb);
                  }
                }
              }
              auto end = high_resolution_clock::now();
              logger_->info(
                  "\t\tThread {} takes {} seconds", tid,
                  duration_cast<duration<double>>(end - start).count());
            } /* lambda */,
            tid /* args to lambda */);
      }
      for (auto& tr : threads) tr.join();
      curr_level.clear();
      // flatten next_level and save to curr_level
      for (const auto& lvl : next_level)
        curr_level.insert(curr_level.end(), lvl.cbegin(), lvl.cend());
      // clear next_level
      for (auto& lvl : next_level) lvl.clear();
      ++lvl_cnt;
    }
  }
};
}  // namespace GDBSCAN

#endif  // GDBSCAN_INCLUDE_SOLVER_H_
