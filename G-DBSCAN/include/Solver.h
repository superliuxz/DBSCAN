//
// Created by William Liu on 2020-01-24.
//

#ifndef GDBSCAN_INCLUDE_SOLVER_H_
#define GDBSCAN_INCLUDE_SOLVER_H_

#include <fstream>
#include <memory>

#include "Graph.h"
#include "Point.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace GDBSCAN {

template <class PointType>
class Solver {
 public:
  explicit Solver(std::unique_ptr<std::ifstream> in, size_t num_nodes,
                  uint min_pts, float radius)
      : num_nodes_(num_nodes), min_pts_(min_pts) {
#if defined(SQRE_RADIUS)
    radius_ = radius * radius;
#else
    radius_ = radius;
#endif
    ifs_ = std::move(in);
    logger_ = spdlog::get("console");
    if (logger_ == nullptr) {
      throw std::runtime_error("logger not created!");
    }

    using namespace std::chrono;
    high_resolution_clock::time_point start = high_resolution_clock::now();

    dataset_ =
        std::make_unique<std::vector<PointType>>(num_nodes_, PointType());
    size_t n;
    float x, y;
    if (std::is_same_v<PointType, point::EuclideanTwoD>) {
      while (*ifs_ >> n >> x >> y) {
        (*dataset_)[n] = PointType(x, y);
      }
    } else {
      throw std::runtime_error("PointType not supported!");
    }
    ifs_->close();

    duration<double> time_spent =
        duration_cast<duration<double>>(high_resolution_clock::now() - start);
    logger_->info("reading vertices takes {} seconds", time_spent.count());
  }

#if defined(TESTING)
  const std::vector<PointType> dataset_view() const { return *dataset_; }
#endif

  const Graph& graph_view() const { return *graph_; }

  /*
   * For each two nodes, if the distance is <= |radius_|, insert them into the
   * graph (|temp_adjacency_list_|). Part of Algorithm 1 (Andrade et al).
   */
  void insert_edges() {
    using namespace std::chrono;
    high_resolution_clock::time_point start = high_resolution_clock::now();

    if (ifs_->is_open()) {
      throw std::runtime_error(
          "Input file stream still open (should not happen)!");
    }
    if (dataset_ == nullptr) {
      throw std::runtime_error("Call prepare_dataset to generate the dataset!");
    }

    graph_ = std::make_unique<Graph>(num_nodes_);
#if defined(BIT_ADJ_UNROLL)  // must apply with BIT_ADJ
    logger_->info("insert_edges - BIT_ADJ_UNROLL");
    size_t N = num_nodes_ / 64u + (num_nodes_ % 64u != 0);
    for (size_t u = 0; u < num_nodes_; ++u) {
      const PointType& upoint = (*dataset_)[u];
      for (size_t outer = 0; outer < N; outer += 4) {
        for (size_t inner = 0; inner < 64; ++inner) {
          size_t v1 = outer * 64llu + inner;
          size_t v2 = v1 + 64;
          size_t v3 = v2 + 64;
          size_t v4 = v3 + 64;
          uint64_t msk = 1llu << inner;
          if (u != v1 && v1 < num_nodes_ && upoint - (*dataset_)[v1] <= radius_)
            graph_->temp_adj_list_[u][outer] |= msk;
          if (u != v2 && v2 < num_nodes_ && upoint - (*dataset_)[v2] <= radius_)
            graph_->temp_adj_list_[u][outer + 1] |= msk;
          if (u != v3 && v3 < num_nodes_ && upoint - (*dataset_)[v3] <= radius_)
            graph_->temp_adj_list_[u][outer + 2] |= msk;
          if (u != v4 && v4 < num_nodes_ && upoint - (*dataset_)[v4] <= radius_)
            graph_->temp_adj_list_[u][outer + 3] |= msk;
        }
      }
    }
#elif defined(BITSET_ADJ_UNROLL)  // must apply with BITSET_ADJ
    logger_->info("insert_edges - BITSET_ADJ_UNROLL");
    size_t N = num_nodes_ / 64u + (num_nodes_ % 64u != 0);
    for (size_t u = 0; u < num_nodes_; ++u) {
      const PointType& upoint = (*dataset_)[u];
      for (size_t outer = 0; outer < N; outer += 4) {
        for (size_t inner = 0; inner < 64; ++inner) {
          size_t v1 = outer * 64llu + inner;
          size_t v2 = v1 + 64;
          size_t v3 = v2 + 64;
          size_t v4 = v3 + 64;
          if (u != v1 && v1 < num_nodes_ && upoint - (*dataset_)[v1] <= radius_)
            graph_->temp_adj_list_[u][outer].set(inner);
          if (u != v2 && v2 < num_nodes_ && upoint - (*dataset_)[v2] <= radius_)
            graph_->temp_adj_list_[u][outer + 1].set(inner);
          if (u != v3 && v3 < num_nodes_ && upoint - (*dataset_)[v3] <= radius_)
            graph_->temp_adj_list_[u][outer + 2].set(inner);
          if (u != v4 && v4 < num_nodes_ && upoint - (*dataset_)[v4] <= radius_)
            graph_->temp_adj_list_[u][outer + 3].set(inner);
        }
      }
    }
#elif defined(TILING)
    logger_->info("insert_edges - TILING");
    // 8k bytes __should__ fit most of modern CPU's L1 dcache.
    const size_t block_size = 8192u / PointType::size();
    logger_->info(
        "PointType size: {} bytes; "
        "block_size: {}",
        PointType::size(), block_size);

#if defined(TRIA_ENUM)  // triangle enumeration
    logger_->info("insert_edges - TRIA_ENUM");
    for (size_t u = 0; u < num_nodes_; u += block_size) {
      size_t uu = std::min(u + block_size, num_nodes_);
      for (size_t v = u + 1; v < num_nodes_; ++v) {
        size_t uuu = std::min(uu, v);
        const PointType& vpoint = (*dataset_)[v];

#if defined(UNROLL_INSE)
        logger_->debug("insert_edges - UNROLL_INSE");
        for (size_t i = u; i < uuu; i += 4) {
          if ((*dataset_)[i] - vpoint <= radius_) {
            graph_->insert_edge(v, i);
          }
          if (i + 1 < uuu && (*dataset_)[i + 1] - vpoint <= radius_) {
            graph_->insert_edge(v, i + 1);
          }
          if (i + 2 < uuu && (*dataset_)[i + 2] - vpoint <= radius_) {
            graph_->insert_edge(v, i + 2);
          }
          if (i + 3 < uuu && (*dataset_)[i + 3] - vpoint <= radius_) {
            graph_->insert_edge(v, i + 3);
          }
        }
#else
        logger_->debug("insert_edges - default");
        for (size_t i = u; i < uuu; ++i) {
          if ((*dataset_)[i] - vpoint <= radius_) {
            graph_->insert_edge(v, i);
          }
        }
#endif
      }
    }
#else  // square enumeration
    logger_->info("insert_edges - square enumeration");
    for (size_t u = 0; u < num_nodes_; u += block_size) {
      size_t uu = std::min(u + block_size, num_nodes_);
      for (size_t v = 0; v < num_nodes_; ++v) {
        const PointType& vpoint = (*dataset_)[v];

#if defined(UNROLL_INSE)
        logger_->debug("insert_edges - UNROLL_INSE");
        for (size_t i = u; i < uu; i += 4) {
          if (i != v && (*dataset_)[i] - vpoint <= radius_) {
            graph_->insert_edge(v, i);
          }
          if (i + 1 < num_nodes_ && i + 1 != v &&
              (*dataset_)[i + 1] - vpoint <= radius_) {
            graph_->insert_edge(v, i + 1);
          }
          if (i + 2 < num_nodes_ && i + 2 != v &&
              (*dataset_)[i + 2] - vpoint <= radius_) {
            graph_->insert_edge(v, i + 2);
          }
          if (i + 3 < num_nodes_ && i + 3 != v &&
              (*dataset_)[i + 3] - vpoint <= radius_) {
            graph_->insert_edge(v, i + 3);
          }
        }
#else
        logger_->debug("insert_edges - default");
        for (size_t i = u; i < uu; ++i) {
          if (i != v && (*dataset_)[i] - vpoint <= radius_) {
            graph_->insert_edge(v, i);
          }
        }
#endif
      }
    }
#endif  // SQRE_ENUM

#else                   // no tiling
    logger_->info("insert_edges - default");
#if defined(TRIA_ENUM)  // triangle enumeration
    logger_->info("insert_edges - TRIA_ENUM");
    for (size_t u = 0; u < num_nodes_; ++u) {
      const PointType& upoint = (*dataset_)[u];
#if defined(UNROLL_INSE)
      logger_->debug("insert_edges - UNROLL_INSE");
      for (size_t v = u + 1; v < num_nodes_; v += 4) {
        if (upoint - (*dataset_)[v] <= radius_) {
          graph_->insert_edge(u, v);
        }
        if (v + 1 < num_nodes_ && upoint - (*dataset_)[v + 1] <= radius_) {
          graph_->insert_edge(u, v + 1);
        }
        if (v + 2 < num_nodes_ && upoint - (*dataset_)[v + 2] <= radius_) {
          graph_->insert_edge(u, v + 2);
        }
        if (v + 3 < num_nodes_ && upoint - (*dataset_)[v + 3] <= radius_) {
          graph_->insert_edge(u, v + 3);
        }
      }
#else
      for (size_t v = u + 1; v < num_nodes_; ++v) {
        if (upoint - (*dataset_)[v] <= radius_) {
          graph_->insert_edge(u, v);
        }
      }
#endif
    }
#else  // square enumeration
    logger_->info("insert_edges - square enumeration");
    for (size_t u = 0; u < num_nodes_; ++u) {
      const PointType& upoint = (*dataset_)[u];
#if defined(UNROLL_INSE)
      logger_->debug("insert_edges - UNROLL_INSE");
      for (size_t v = 0; v < num_nodes_; v += 4) {
        if (u != v && upoint - (*dataset_)[v] <= radius_) {
          graph_->insert_edge(u, v);
        }
        if (u != v + 1 && v + 1 < num_nodes_ &&
            upoint - (*dataset_)[v + 1] <= radius_) {
          graph_->insert_edge(u, v + 1);
        }
        if (u != v + 2 && v + 2 < num_nodes_ &&
            upoint - (*dataset_)[v + 2] <= radius_) {
          graph_->insert_edge(u, v + 2);
        }
        if (u != v + 3 && v + 3 < num_nodes_ &&
            upoint - (*dataset_)[v + 3] <= radius_) {
          graph_->insert_edge(u, v + 3);
        }
      }
#else
      for (size_t v = 0; v < num_nodes_; ++v) {
        if (u != v && upoint - (*dataset_)[v] <= radius_) {
          graph_->insert_edge(u, v);
        }
      }
#endif
    }
#endif  // TRIA_ENUM

#endif  // TILING
    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double> time_spent = duration_cast<duration<double>>(end - start);
    logger_->info(
        "insert_edges (Algorithm 1) - graph_->insert_edge takes {} seconds",
        time_spent.count());
  }

  /*
   * Construct |Va| and |Ea| from |temp_adjacency_list_|. Part of Algorithm 1
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
      logger_->debug("{} has {} neighbours within {}", node,
                     graph_->Va[node * 2], radius_);
      logger_->debug("{} >= {}: {}", graph_->Va[node * 2], min_pts_,
                     graph_->Va[node * 2] >= min_pts_ ? "true" : "false");
      if (graph_->Va[node * 2] >= min_pts_) {
        logger_->debug("{} to Core", node);
        graph_->membership[node] = membership::Core;
      } else {
        logger_->debug("{} to Noise", node);
        graph_->membership[node] = membership::Noise;
      }
    }

    duration<double> time_spent =
        duration_cast<duration<double>>(high_resolution_clock::now() - start);
    logger_->info("classify_nodes takes {} seconds", time_spent.count());
  }

  /*
   * Algorithm 2 (BFS/DFS) in Andrade et al.
   */
  void identify_cluster() const {
    using namespace std::chrono;
    high_resolution_clock::time_point start = high_resolution_clock::now();

    int cluster = 0;
    for (size_t node = 0; node < num_nodes_; ++node) {
      if (graph_->cluster_ids[node] == -1 &&
          graph_->membership[node] == membership::Core) {
        graph_->cluster_ids[node] = cluster;
#if defined(DFS)
        logger_->debug("start dfs on node {} with cluster {}", node, cluster);
        dfs(node, cluster);
#else
        logger_->debug("start bfs on node {} with cluster {}", node, cluster);
        bfs(node, cluster);
#endif
        ++cluster;
      }
    }

    duration<double> time_spent =
        duration_cast<duration<double>>(high_resolution_clock::now() - start);
    logger_->info("identify_cluster (Algorithm 2) takes {} seconds",
                  time_spent.count());
  }

 private:
  size_t num_nodes_;
  size_t min_pts_;
  float radius_;
  std::unique_ptr<std::vector<PointType>> dataset_ = nullptr;
  std::unique_ptr<Graph> graph_ = nullptr;
  std::unique_ptr<std::ifstream> ifs_ = nullptr;
  std::shared_ptr<spdlog::logger> logger_ = nullptr;

  /*
   * BFS. Start from |node| and visit all the reachable neighbours. If a
   * neighbour is Noise, relabel it to Border.
   */
  void bfs(size_t node, int cluster) const {
    std::vector<size_t> q{node};
    std::vector<size_t> next_level;
    while (!q.empty()) {
      for (const size_t& curr : q) {
        logger_->debug("visiting node {}", curr);
        // Relabel a reachable Noise node, but do not keep exploring.
        if (graph_->membership[curr] == membership::Noise) {
          logger_->debug("\tnode {} is relabeled from Noise to Border", curr);
          graph_->membership[curr] = membership::Border;
          continue;
        }

        size_t num_neighbours = graph_->Va[2 * curr];
        size_t start_pos = graph_->Va[2 * curr + 1];
#if defined(UNROLL_BFS)
        for (size_t i = 0; i < num_neighbours; i += 4) {
          if (i < num_neighbours) {
            size_t nb0 = graph_->Ea[start_pos + i];
            if (graph_->cluster_ids[nb0] == -1) {
              // cluster the node
              logger_->debug("\tnode {} is clustered tp {}", nb0, cluster);
              graph_->cluster_ids[nb0] = cluster;
              logger_->debug("\tneighbour {} of node {} is queued", nb0, curr);
              next_level.emplace_back(nb0);
            }
          }
          if (i + 1 < num_neighbours) {
            size_t nb1 = graph_->Ea[start_pos + i + 1];
            if (graph_->cluster_ids[nb1] == -1) {
              // cluster the node
              logger_->debug("\tnode {} is clustered tp {}", nb1, cluster);
              graph_->cluster_ids[nb1] = cluster;
              logger_->debug("\tneighbour {} of node {} is queued", nb1, curr);
              next_level.emplace_back(nb1);
            }
          }
          if (i + 2 < num_neighbours) {
            size_t nb2 = graph_->Ea[start_pos + i + 2];
            if (graph_->cluster_ids[nb2] == -1) {
              // cluster the node
              logger_->debug("\tnode {} is clustered tp {}", nb2, cluster);
              graph_->cluster_ids[nb2] = cluster;
              logger_->debug("\tneighbour {} of node {} is queued", nb2, curr);
              next_level.emplace_back(nb2);
            }
          }
          if (i + 3 < num_neighbours) {
            size_t nb3 = graph_->Ea[start_pos + i + 3];
            if (graph_->cluster_ids[nb3] == -1) {
              // cluster the node
              logger_->debug("\tnode {} is clustered tp {}", nb3, cluster);
              graph_->cluster_ids[nb3] = cluster;
              logger_->debug("\tneighbour {} of node {} is queued", nb3, curr);
              next_level.emplace_back(nb3);
            }
          }
        }
#else
        for (size_t i = 0; i < num_neighbours; ++i) {
          size_t nb = graph_->Ea[start_pos + i];
          if (graph_->cluster_ids[nb] == -1) {
            // cluster the node
            logger_->debug("\tnode {} is clustered tp {}", nb, cluster);
            graph_->cluster_ids[nb] = cluster;
            logger_->debug("\tneighbour {} of node {} is queued", nb, curr);
            next_level.emplace_back(nb);
          }
        }
#endif
      }
      q = std::move(next_level);
      next_level.clear();
    }
  }

  /*
   * DFS version.
   */
  void dfs(size_t node, int cluster) const {
    std::vector<size_t> stack{node};
    while (!stack.empty()) {
      size_t curr = stack.back();
      stack.pop_back();

      logger_->debug("visiting node {}", curr);
      // Relabel a reachable Noise node, but do not keep exploring.
      if (graph_->membership[curr] == membership::Noise) {
        logger_->debug("\tnode {} is relabeled from Noise to Border", curr);
        graph_->membership[curr] = membership::Border;
        continue;
      }

      size_t num_neighbours = graph_->Va[2 * curr];
      size_t start_pos = graph_->Va[2 * curr + 1];

      for (size_t i = 0; i < num_neighbours; ++i) {
        size_t nb = graph_->Ea[start_pos + i];
        if (graph_->cluster_ids[nb] == -1) {
          // cluster the node
          logger_->debug("\tnode {} is clustered tp {}", nb, cluster);
          graph_->cluster_ids[nb] = cluster;
          logger_->debug("\tneighbour {} of node {} is stacked", nb, curr);
          stack.emplace_back(nb);
        }
      }
    }
  }
};

template <class PointType>
static std::unique_ptr<Solver<PointType>> make_solver(std::string input,
                                                      uint min_pts,
                                                      float radius) {
  size_t num_nodes;
  auto ifs = std::make_unique<std::ifstream>(input);
  *ifs >> num_nodes;
  return std::make_unique<Solver<PointType>>(std::move(ifs), num_nodes, min_pts,
                                             radius);
}
}  // namespace GDBSCAN

#endif  // GDBSCAN_INCLUDE_SOLVER_H_
