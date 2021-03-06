//
// Created by William Liu on 2020-03-18.
//

#include "solver.h"

#if defined(AVX)
#include <nmmintrin.h>
#endif

#include <cmath>
#include <limits>
#include <memory>
#include <thread>

#include "dataset.h"
#include "graph.h"
#include "spdlog/spdlog.h"

// ctor
DBSCAN::Solver::Solver(const std::string& input, const uint64_t min_pts,
                       const float radius, const uint8_t num_threads)
    : min_pts_(min_pts),
      squared_radius_(radius * radius),
      num_threads_(num_threads) {
  logger_ = spdlog::get("console");
  if (logger_ == nullptr) {
    throw std::runtime_error("logger not created!");
  }
#if defined(AVX)
  sq_rad8_ = _mm256_set1_ps(squared_radius_);
#endif
  using namespace std::chrono;
  high_resolution_clock::time_point start = high_resolution_clock::now();

  auto ifs = std::ifstream(input);
  ifs >> num_vtx_;
  dataset_ = std::make_unique<DBSCAN::input_type::TwoDimPoints>(num_vtx_);
  uint64_t n;
  float x, y;
  // grid
  float max_x = std::numeric_limits<float>::min(),
        max_y = std::numeric_limits<float>::min(),
        min_x = std::numeric_limits<float>::max(),
        min_y = std::numeric_limits<float>::max();
  while (ifs >> n >> x >> y) {
    dataset_->d1[n] = x;
    dataset_->d2[n] = y;
    // manually offset by radius/2 such the min/max values fall within
    // second/second last cell.
    max_x = std::max(max_x, x + radius / 2);
    min_x = std::min(min_x, x - radius / 2);
    max_y = std::max(max_y, y + radius / 2);
    min_y = std::min(min_y, y - radius / 2);
  }

  duration<double> time_spent =
      duration_cast<duration<double>>(high_resolution_clock::now() - start);
  logger_->info("reading vertices takes {} seconds", time_spent.count());

  cluster_ids.resize(num_vtx_, -1);
  memberships.resize(num_vtx_, DBSCAN::membership::Noise);
  grid_ = std::make_unique<Grid>(max_x, max_y, min_x, min_y, radius, num_vtx_,
                                 num_threads_);
}

void DBSCAN::Solver::InsertEdges() {
  using namespace std::chrono;
  high_resolution_clock::time_point start = high_resolution_clock::now();

  if (dataset_ == nullptr) {
    throw std::runtime_error("Call prepare_dataset to generate the dataset!");
  }

  graph_ = std::make_unique<Graph>(num_vtx_, num_threads_);

  std::vector<std::thread> threads(num_threads_);
#if defined(BIT_ADJ)
  logger_->info("InsertEdges - BIT_ADJ");
  const uint64_t N = std::ceil(num_vtx_ / 64.f);
  for (uint8_t tid = 0; tid < num_threads_; ++tid) {
    threads[tid] = std::thread(
        [this, N](const uint8_t tid) {
          auto t0 = high_resolution_clock::now();
          for (uint64_t u = tid; u < num_vtx_; u += num_threads_) {
            const float &ux = dataset_->d1[u], uy = dataset_->d2[u];
#if defined(AVX)
            const __m256 u_x8 = _mm256_set1_ps(ux);
            const __m256 u_y8 = _mm256_set1_ps(uy);
            for (uint64_t outer = 0; outer < N; ++outer) {
              for (uint64_t inner = 0; inner < 64; inner += 8) {
                const uint64_t v0 = outer * 64llu + inner;
                const uint64_t v1 = v0 + 1;
                const uint64_t v2 = v0 + 2;
                const uint64_t v3 = v0 + 3;
                const uint64_t v4 = v0 + 4;
                const uint64_t v5 = v0 + 5;
                const uint64_t v6 = v0 + 6;
                const uint64_t v7 = v0 + 7;
                // TODO: if num_vtx_ is not a multiple of 8
                // logger_->trace("vertex {} (num_vtx_ {}); outer{}; inner
                // {}", u, num_vtx_, outer, inner);

                const float* const v_x_ptr = &(dataset_->d1.front());
                const __m256 v_x_8 = _mm256_load_ps(v_x_ptr + v0);
                const float* const v_y_ptr = &(dataset_->d2.front());
                const __m256 v_y_8 = _mm256_load_ps(v_y_ptr + v0);
                const __m256 x_diff_8 = _mm256_sub_ps(u_x8, v_x_8);
                const __m256 x_diff_sq_8 = _mm256_mul_ps(x_diff_8, x_diff_8);
                const __m256 y_diff_8 = _mm256_sub_ps(u_y8, v_y_8);
                const __m256 y_diff_sq_8 = _mm256_mul_ps(y_diff_8, y_diff_8);
                const __m256 sum = _mm256_add_ps(x_diff_sq_8, y_diff_sq_8);

                // const auto temp = reinterpret_cast<float const*>(&sum);
                // logger_->trace("summation of X^2 and Y^2 (sum):");
                // for (uint64_t i = 0; i < 8; ++i)
                //   logger_->trace("\t{}", temp[i]);

                const int cmp = _mm256_movemask_ps(
                    _mm256_cmp_ps(sum, sq_rad8_, _CMP_LE_OS));
                // logger_->trace(
                //     "comparison of X^2+Y^2 against radius^2 (cmp): {}",
                //     cmp);

                if (u != v0 && v0 < num_vtx_ && (cmp & 1 << 0))
                  graph_->InsertEdge(u, outer, 1llu << inner);
                if (u != v1 && v1 < num_vtx_ && (cmp & 1 << 1))
                  graph_->InsertEdge(u, outer, 1llu << (inner + 1));
                if (u != v2 && v2 < num_vtx_ && (cmp & 1 << 2))
                  graph_->InsertEdge(u, outer, 1llu << (inner + 2));
                if (u != v3 && v3 < num_vtx_ && (cmp & 1 << 3))
                  graph_->InsertEdge(u, outer, 1llu << (inner + 3));
                if (u != v4 && v4 < num_vtx_ && (cmp & 1 << 4))
                  graph_->InsertEdge(u, outer, 1llu << (inner + 4));
                if (u != v5 && v5 < num_vtx_ && (cmp & 1 << 5))
                  graph_->InsertEdge(u, outer, 1llu << (inner + 5));
                if (u != v6 && v6 < num_vtx_ && (cmp & 1 << 6))
                  graph_->InsertEdge(u, outer, 1llu << (inner + 6));
                if (u != v7 && v7 < num_vtx_ && (cmp & 1 << 7))
                  graph_->InsertEdge(u, outer, 1llu << (inner + 7));
              }
            }
#else
            const auto dist =
                input_type::TwoDimPoints::euclidean_distance_square;
            for (uint64_t outer = 0; outer < N; outer += 4) {
              for (uint64_t inner = 0; inner < 64; ++inner) {
                const uint64_t v1 = outer * 64llu + inner;
                const uint64_t v2 = v1 + 64;
                const uint64_t v3 = v2 + 64;
                const uint64_t v4 = v3 + 64;
                const uint64_t msk = 1llu << inner;
                if (u != v1 && v1 < num_vtx_ &&
                    dist(ux, uy, dataset_->d1[v1], dataset_->d2[v1]) <=
                        squared_radius_)
                  graph_->InsertEdge(u, outer, msk);
                if (u != v2 && v2 < num_vtx_ &&
                    dist(ux, uy, dataset_->d1[v2], dataset_->d2[v2]) <=
                        squared_radius_)
                  graph_->InsertEdge(u, outer + 1, msk);
                if (u != v3 && v3 < num_vtx_ &&
                    dist(ux, uy, dataset_->d1[v3], dataset_->d2[v3]) <=
                        squared_radius_)
                  graph_->InsertEdge(u, outer + 2, msk);
                if (u != v4 && v4 < num_vtx_ &&
                    dist(ux, uy, dataset_->d1[v4], dataset_->d2[v4]) <=
                        squared_radius_)
                  graph_->InsertEdge(u, outer + 3, msk);
              }
            }
#endif
          }
          auto t1 = high_resolution_clock::now();
          logger_->info("\tThread {} takes {} seconds", tid,
                        duration_cast<duration<double>>(t1 - t0).count());
        }, /* lambda */
        tid /* args to lambda */);
  }
#else
  logger_->info("InsertEdges - default");
  const auto dist = input_type::TwoDimPoints::euclidean_distance_square;
  for (uint8_t tid = 0; tid < num_threads_; ++tid) {
    threads[tid] = std::thread(
        [this, &dist](const uint8_t tid) {
          auto t0 = high_resolution_clock::now();
#if defined(AVX)
          // each float is 4 bytes; a 256bit register is 32 bytes. Hence 8
          // float at-a-time.
          for (uint64_t u = tid; u < num_vtx_; u += num_threads_) {
            const float &ux = dataset_->d1[u], uy = dataset_->d2[u];
            const __m256 u_x8 = _mm256_set1_ps(ux);
            const __m256 u_y8 = _mm256_set1_ps(uy);
            const std::vector<uint64_t> nbs =
                grid_->GetNeighbouringVtx(u, ux, uy);
            graph_->StartInsert(u, nbs.size());
            for (uint64_t i = 0; i < nbs.size(); i += 8) {
              const __m256 v_x_8 = _mm256_set_ps(
                  dataset_->d1[nbs[i]],
                  i + 1 < nbs.size() ? dataset_->d1[nbs[i + 1]] : max_radius_,
                  i + 2 < nbs.size() ? dataset_->d1[nbs[i + 2]] : max_radius_,
                  i + 3 < nbs.size() ? dataset_->d1[nbs[i + 3]] : max_radius_,
                  i + 4 < nbs.size() ? dataset_->d1[nbs[i + 4]] : max_radius_,
                  i + 5 < nbs.size() ? dataset_->d1[nbs[i + 5]] : max_radius_,
                  i + 6 < nbs.size() ? dataset_->d1[nbs[i + 6]] : max_radius_,
                  i + 7 < nbs.size() ? dataset_->d1[nbs[i + 7]] : max_radius_);
              const __m256 v_y_8 = _mm256_set_ps(
                  dataset_->d2[nbs[i]],
                  i + 1 < nbs.size() ? dataset_->d2[nbs[i + 1]] : max_radius_,
                  i + 2 < nbs.size() ? dataset_->d2[nbs[i + 2]] : max_radius_,
                  i + 3 < nbs.size() ? dataset_->d2[nbs[i + 3]] : max_radius_,
                  i + 4 < nbs.size() ? dataset_->d2[nbs[i + 4]] : max_radius_,
                  i + 5 < nbs.size() ? dataset_->d2[nbs[i + 5]] : max_radius_,
                  i + 6 < nbs.size() ? dataset_->d2[nbs[i + 6]] : max_radius_,
                  i + 7 < nbs.size() ? dataset_->d2[nbs[i + 7]] : max_radius_);

              const __m256 x_diff_8 = _mm256_sub_ps(u_x8, v_x_8);
              const __m256 x_diff_sq_8 = _mm256_mul_ps(x_diff_8, x_diff_8);
              const __m256 y_diff_8 = _mm256_sub_ps(u_y8, v_y_8);
              const __m256 y_diff_sq_8 = _mm256_mul_ps(y_diff_8, y_diff_8);

              const __m256 sum = _mm256_add_ps(x_diff_sq_8, y_diff_sq_8);

              const int cmp =
                  _mm256_movemask_ps(_mm256_cmp_ps(sum, sq_rad8_, _CMP_LE_OS));
              if (cmp & 1 << 7) graph_->InsertEdge(u, nbs[i]);
              if (cmp & 1 << 6) graph_->InsertEdge(u, nbs[i + 1]);
              if (cmp & 1 << 5) graph_->InsertEdge(u, nbs[i + 2]);
              if (cmp & 1 << 4) graph_->InsertEdge(u, nbs[i + 3]);
              if (cmp & 1 << 3) graph_->InsertEdge(u, nbs[i + 4]);
              if (cmp & 1 << 2) graph_->InsertEdge(u, nbs[i + 5]);
              if (cmp & 1 << 1) graph_->InsertEdge(u, nbs[i + 6]);
              if (cmp & 1 << 0) graph_->InsertEdge(u, nbs[i + 7]);
            }
            graph_->FinishInsert(u);
          }
#else
          for (uint64_t u = tid; u < num_vtx_; u += num_threads_) {
            const float &ux = dataset_->d1[u], uy = dataset_->d2[u];
            const std::vector<uint64_t> nbs =
                grid_->GetNeighbouringVtx(u, ux, uy);
            graph_->StartInsert(u, nbs.size());
            // logger_->debug("possible nbs of {}: {}", u,
            //                DBSCAN::utils::print_vector("", nbs));
            for (const auto v : nbs) {
              if (dist(ux, uy, dataset_->d1[v], dataset_->d2[v]) <=
                  squared_radius_)
                graph_->InsertEdge(u, v);
            }
            graph_->FinishInsert(u);
          }
#endif
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
  logger_->info("InsertEdges takes {} seconds", time_spent.count());
}

void DBSCAN::Solver::ClassifyNoises() {
  using namespace std::chrono;
  high_resolution_clock::time_point start = high_resolution_clock::now();
  if (graph_ == nullptr) {
    throw std::runtime_error("Call InsertEdges to generate the graph!");
  }
  for (uint64_t vertex = 0; vertex < num_vtx_; ++vertex) {
    // logger_->trace("{} has {} neighbours within {}", vertex,
    //                graph_->num_nbs[vertex], squared_radius_);
    // logger_->trace("{} >= {}: {}", graph_->num_nbs[vertex], min_pts_,
    //                graph_->num_nbs[vertex] >= min_pts_ ? "true" :
    //                "false");
    if (graph_->num_nbs[vertex] >= min_pts_) {
      // logger_->trace("{} to Core", vertex);
      memberships[vertex] = Core;
    } else {
      // logger_->trace("{} to Noise", vertex);
      memberships[vertex] = Noise;
    }
  }
  duration<double> time_spent =
      duration_cast<duration<double>>(high_resolution_clock::now() - start);
  logger_->info("ClassifyNoises takes {} seconds", time_spent.count());
}

void DBSCAN::Solver::IdentifyClusters() {
  using namespace std::chrono;
  high_resolution_clock::time_point start = high_resolution_clock::now();
  int cluster = 0;
  for (uint64_t vertex = 0; vertex < num_vtx_; ++vertex) {
    if (cluster_ids[vertex] == -1 && memberships[vertex] == Core) {
      cluster_ids[vertex] = cluster;
      // logger_->debug("start bfs on vertex {} with cluster {}", vertex,
      // cluster);
      BFS_(vertex, cluster);
      ++cluster;
    }
  }
  duration<double> time_spent =
      duration_cast<duration<double>>(high_resolution_clock::now() - start);
  logger_->info("IdentifyClusters takes {} seconds", time_spent.count());
}

void DBSCAN::Solver::BFS_(const uint64_t start_vertex, const int cluster) {
  std::vector<uint64_t> curr_level{start_vertex};
  // each thread has its own partial frontier.
  std::vector<std::vector<uint64_t>> next_level(num_threads_,
                                                std::vector<uint64_t>());

  std::vector<std::thread> threads(num_threads_);
  // uint64_t lvl_cnt = 0;
  while (!curr_level.empty()) {
    // logger_->info("\tBFS level {}", lvl_cnt);
    for (uint8_t tid = 0u; tid < num_threads_; ++tid) {
      threads[tid] = std::thread(
          [this, &curr_level, &next_level, cluster](const uint8_t tid) {
            // using namespace std::chrono;
            // auto p_t0 = high_resolution_clock::now();
            for (uint64_t curr_vertex_idx = tid;
                 curr_vertex_idx < curr_level.size();
                 curr_vertex_idx += num_threads_) {
              uint64_t vertex = curr_level[curr_vertex_idx];
              // logger_->trace("visiting vertex {}", vertex);
              // Relabel a reachable Noise vertex, but do not keep exploring.
              if (memberships[vertex] == Noise) {
                // logger_->trace("\tvertex {} is relabeled from Noise to
                // Border", vertex);
                memberships[vertex] = Border;
                continue;
              }
              const uint64_t start_pos = graph_->start_pos[vertex];
              const uint64_t num_neighbours = graph_->num_nbs[vertex];
              for (uint64_t i = 0; i < num_neighbours; ++i) {
                uint64_t nb = graph_->neighbours[start_pos + i];
                if (cluster_ids[nb] == -1) {
                  // cluster the vertex
                  // logger_->trace("\tvertex {} is clustered to {}", nb,
                  // cluster);
                  cluster_ids[nb] = cluster;
                  // logger_->trace("\tneighbour {} of vertex {} is queued", nb,
                  // vertex);
                  next_level[tid].emplace_back(nb);
                }
              }
            }
            // auto p_t1 = high_resolution_clock::now();
            // logger_->info(
            //     "\t\tThread {} takes {} seconds", tid,
            //     duration_cast<duration<double>>(p_t1 - p_t0).count());
          } /* lambda */,
          tid /* args to lambda */);
    }
    for (auto& tr : threads) tr.join();
    curr_level.clear();
    // sync barrier
    // flatten next_level and save to curr_level
    for (const auto& lvl : next_level)
      curr_level.insert(curr_level.end(), lvl.cbegin(), lvl.cend());
    // clear next_level
    for (auto& lvl : next_level) lvl.clear();
    // ++lvl_cnt;
  }
}
