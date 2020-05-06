//
// Created by will on 2020-03-26.
//

#include <thrust/count.h>

#include <fstream>

#include "DBSCAN/membership.h"
#include "gdbscan.cuh"
#include "gdbscan_kernel_functions.cuh"

// https://stackoverflow.com/a/14038590
#define CUDA_ERR_CHK(code) \
  { cuda_err_chk((code), __FILE__, __LINE__); }
inline void cuda_err_chk(cudaError_t code, const char *file, int line,
                         bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "\tCUDA ERROR: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

GDBSCAN::Solver::Solver(const std::string &input, const uint32_t min_pts,
                        const float radius)
    : radius_(radius), min_pts_(min_pts) {
  auto ifs = std::ifstream(input);
  ifs >> num_vtx_;
  num_blocks_ = std::ceil(num_vtx_ / static_cast<float>(BLOCK_SIZE));
  x_.resize(num_vtx_, 0);
  y_.resize(num_vtx_, 0);
  l1norm_.resize(num_vtx_, 0);
  vtx_mapper_.resize(num_vtx_, 0);
  memberships.resize(num_vtx_, DBSCAN::membership::Noise);
  cluster_ids.resize(num_vtx_, -1);

  size_t n;
  float x, y;
  while (ifs >> n >> x >> y) {
    x_[n] = x;
    y_[n] = y;
    l1norm_[n] = std::abs(x) + std::abs(y);
    vtx_mapper_[n] = n;
  }
#if defined(DBSCAN_TESTING)
  start_pos.resize(num_vtx_, 0);
  num_neighbours.resize(num_vtx_, 0);
#endif
}

void GDBSCAN::Solver::sort_input_by_l1norm() {
  const auto N = sizeof(x_[0]) * num_vtx_;
  const auto K = sizeof(dev_vtx_mapper_[0]) * num_vtx_;
  printf("sort_input_by_l1norm needs: %lf MB\n",
         static_cast<double>(N * 3 + K) / 1024.f / 1024.f);

  CUDA_ERR_CHK(cudaMalloc((void **)&dev_x_, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_y_, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_l1norm_, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_vtx_mapper_, K));
  CUDA_ERR_CHK(cudaMemcpy(dev_x_, thrust::raw_pointer_cast(x_.data()), N, H2D));
  CUDA_ERR_CHK(cudaMemcpy(dev_y_, thrust::raw_pointer_cast(y_.data()), N, H2D));
  CUDA_ERR_CHK(cudaMemcpy(dev_l1norm_, thrust::raw_pointer_cast(l1norm_.data()),
                          N, H2D));
  CUDA_ERR_CHK(cudaMemcpy(
      dev_vtx_mapper_, thrust::raw_pointer_cast(vtx_mapper_.data()), K, H2D));

  // https://thrust.github.io/doc/classthrust_1_1zip__iterator.html
  typedef typename thrust::tuple<float *, float *, uint32_t *> IteratorTuple;
  typedef typename thrust::zip_iterator<IteratorTuple> ZipIterator;
  ZipIterator begin(thrust::make_tuple(dev_x_, dev_y_, dev_vtx_mapper_));
  thrust::sort_by_key(thrust::device, dev_l1norm_, dev_l1norm_ + num_vtx_,
                      begin);
#if defined(DBSCAN_TESTING)
  CUDA_ERR_CHK(cudaMemcpy(thrust::raw_pointer_cast(x_.data()), dev_x_, N, D2H));
  CUDA_ERR_CHK(cudaMemcpy(thrust::raw_pointer_cast(y_.data()), dev_y_, N, D2H));
  CUDA_ERR_CHK(cudaMemcpy(thrust::raw_pointer_cast(l1norm_.data()), dev_l1norm_,
                          N, D2H));
  CUDA_ERR_CHK(cudaMemcpy(thrust::raw_pointer_cast(vtx_mapper_.data()),
                          dev_vtx_mapper_, K, D2H));
//  printf("sorted x: ");
//  for (auto & v : x_) printf("%f ", v);
//  printf("\nsorted y: ");
//  for (auto & v : y_) printf("%f ", v);
//  printf("\nsorted l1norm: ");
//  for (auto & v : l1norm_) printf("%f ", v);
//  printf("\nsorted vtx_mapper: ");
//  for (auto & v : vtx_mapper_) printf("%lu ", v);
//  printf("\n");
#endif
  // dev_x_, dev_y_, dev_l1norm_, dev_vtx_mapper_ in GPU RAM.
}

void GDBSCAN::Solver::calc_num_neighbours() {
  const auto N = sizeof(x_[0]) * num_vtx_;
  const auto K = sizeof(dev_num_neighbours_[0]) * num_vtx_;

  printf("calc_num_neighbours needs: %lf MB\n",
         static_cast<double>(N * 3 + K * 2) / 1024.f / 1024.f);

  uint32_t last_vtx_num_nbs = 0;
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_num_neighbours_, K));

  GDBSCAN::kernel_functions::k_num_nbs<<<num_blocks_, BLOCK_SIZE>>>(
      dev_x_, dev_y_, dev_l1norm_, dev_vtx_mapper_, radius_, num_vtx_,
      dev_num_neighbours_);
  CUDA_ERR_CHK(cudaPeekAtLastError());
  CUDA_ERR_CHK(cudaMemcpy(&last_vtx_num_nbs, dev_num_neighbours_ + num_vtx_ - 1,
                          sizeof(last_vtx_num_nbs), D2H));
  total_num_nbs_ += last_vtx_num_nbs;
#if defined(DBSCAN_TESTING)
  CUDA_ERR_CHK(cudaMemcpy(thrust::raw_pointer_cast(num_neighbours.data()),
                          dev_num_neighbours_, K, D2H));
//  printf("num_nbs: ");
//  for (auto &v : num_neighbours) printf("%u ", v);
//  printf("\n");
#endif
  // dev_x_, dev_y_, dev_l1norm_, dev_vtx_mapper_, dev_num_neighbours_
  // in GPU RAM.
}

void GDBSCAN::Solver::calc_start_pos() {
  uint32_t last_vtx_start_pos = 0;

  const auto N = sizeof(dev_start_pos_[0]) * num_vtx_;
  const auto K = sizeof(dev_num_neighbours_[0]) * num_vtx_;

  printf("calc_start_pos needs: %lf MB\n",
         static_cast<double>(N * 3 + K * 3) / 1024.f / 1024.f);
  // Do not free dev_start_pos_. It's required for the rest of algorithm.
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_start_pos_, N));
  thrust::exclusive_scan(thrust::device, dev_num_neighbours_,
                         dev_num_neighbours_ + num_vtx_, dev_start_pos_);
  CUDA_ERR_CHK(cudaMemcpy(&last_vtx_start_pos, dev_start_pos_ + num_vtx_ - 1,
                          sizeof(uint32_t), D2H));
  total_num_nbs_ += last_vtx_start_pos;
#if defined(DBSCAN_TESTING)
  CUDA_ERR_CHK(cudaMemcpy(thrust::raw_pointer_cast(start_pos.data()),
                          dev_start_pos_, N, D2H));
//  printf("start_pos: ");
//  for (auto & v : start_pos) printf("%lu ", v);
//  printf("\n");
#endif
  // dev_x_, dev_y_, dev_l1norm_, dev_vtx_mapper_, dev_num_neighbours_,
  // dev_start_pos_, in GPU RAM.
}

void GDBSCAN::Solver::append_neighbours() {
#if defined(DBSCAN_TESTING)
  neighbours.resize(total_num_nbs_, 0);
#endif
  //  printf("size of neighbours array: %lu\n", total_num_nbs_);

  const auto N = sizeof(x_[0]) * num_vtx_;
  const auto K = sizeof(dev_start_pos_[0]) * num_vtx_;
  const auto J = sizeof(dev_neighbours_[0]) * total_num_nbs_;

  printf("append_neighbours needs: %lf MB\n",
         static_cast<double>(N * 3 + K * 3 + J) / 1024.f / 1024.f);

  CUDA_ERR_CHK(cudaMalloc((void **)&dev_neighbours_, J));

  GDBSCAN::kernel_functions::k_append_neighbours<<<num_blocks_, BLOCK_SIZE>>>(
      dev_x_, dev_y_, dev_l1norm_, dev_vtx_mapper_, dev_start_pos_, radius_,
      num_vtx_, dev_neighbours_);
  CUDA_ERR_CHK(cudaPeekAtLastError());

  // dev_x_ and dev_y_ are no longer used.
  CUDA_ERR_CHK(cudaFree(dev_x_));
  CUDA_ERR_CHK(cudaFree(dev_y_));
  // graph has been fully constructed, hence free all the sorting related.
  CUDA_ERR_CHK(cudaFree(dev_l1norm_));
  CUDA_ERR_CHK(cudaFree(dev_vtx_mapper_));
#if defined(DBSCAN_TESTING)
  CUDA_ERR_CHK(cudaMemcpy(thrust::raw_pointer_cast(neighbours.data()),
                          dev_neighbours_, J, D2H));
//  printf("neighbours: ");
//  for (auto &v : neighbours) printf("%lu ", v);
//  printf("\n");
#endif
  // dev_num_neighbours_, dev_start_pos_, dev_neighbours_ in GPU RAM.
}

void GDBSCAN::Solver::identify_cores() {
  const auto N = sizeof(dev_num_neighbours_[0]) * num_vtx_;
  const auto M = sizeof(dev_membership_[0]) * num_vtx_;
  const auto J = sizeof(dev_neighbours_[0]) * total_num_nbs_;
  printf("identify_cores needs: %lf MB\n",
         static_cast<double>(N * 2 + J + M) / 1024.f / 1024.f);
  // Do not free dev_membership_ as it's used in BFS. The content will be
  // copied out at the end of |identify_clusters|.
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_membership_, M));
  GDBSCAN::kernel_functions::k_identify_cores<<<num_blocks_, BLOCK_SIZE>>>(
      dev_num_neighbours_, dev_membership_, num_vtx_, min_pts_);
  // Copy the membership data from GPU to CPU RAM as needed for the BFS
  // condition check.
  CUDA_ERR_CHK(cudaMemcpy(thrust::raw_pointer_cast(memberships.data()),
                          dev_membership_, M, D2H));
  // dev_num_neighbours_, dev_start_pos_, dev_neighbours_, dev_membership_ in
  // GPU RAM.
}

void GDBSCAN::Solver::identify_clusters() {
  const auto T = sizeof(bool) * num_vtx_;
  const auto N = sizeof(dev_num_neighbours_[0]) * num_vtx_;
  const auto L = sizeof(dev_membership_[0]) * num_vtx_;
  const auto K = sizeof(dev_neighbours_[0]) * total_num_nbs_;

  printf("identify_clusters needs: %lf MB\n",
         static_cast<double>(N * 2 + K + L + T * 2) / 1024.f / 1024.f);

  int cluster = 0;
  for (uint32_t u = 0; u < num_vtx_; ++u) {
    if (cluster_ids[u] == -1 && memberships[u] == DBSCAN::membership::Core) {
      //      printf("vtx %lu cluster %d\n", u, cluster);
      bfs(u, cluster);
      ++cluster;
    }
  }
  CUDA_ERR_CHK(cudaFree(dev_num_neighbours_));
  CUDA_ERR_CHK(cudaFree(dev_start_pos_));
  CUDA_ERR_CHK(cudaFree(dev_neighbours_));
  CUDA_ERR_CHK(cudaFree(dev_membership_));
  // nothing in GPU RAM.
}

void GDBSCAN::Solver::bfs(const uint32_t u, const int cluster) {
  auto visited = new bool[num_vtx_]();
  auto frontier = new bool[num_vtx_]();
  uint32_t num_frontier = 1;
  frontier[u] = true;
  const auto T = sizeof(visited[0]) * num_vtx_;
  const auto L = sizeof(*dev_membership_) * num_vtx_;

  bool *dev_visited, *dev_frontier;
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_visited, T));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_frontier, T));
  CUDA_ERR_CHK(cudaMemcpy(dev_visited, visited, T, H2D));
  CUDA_ERR_CHK(cudaMemcpy(dev_frontier, frontier, T, H2D));
  CUDA_ERR_CHK(cudaMemcpy(
      dev_membership_, thrust::raw_pointer_cast(memberships.data()), L, H2D));

  while (num_frontier > 0) {
    //    printf("\tnumber_frontier: %lu\n", num_frontier);
    GDBSCAN::kernel_functions::k_bfs<<<num_blocks_, BLOCK_SIZE>>>(
        dev_visited, dev_frontier, dev_num_neighbours_, dev_start_pos_,
        dev_neighbours_, dev_membership_, num_vtx_);
    CUDA_ERR_CHK(cudaPeekAtLastError());
    num_frontier = thrust::count(thrust::device, dev_frontier,
                                 dev_frontier + num_vtx_, true);
  }
  // we don't care about he content in dev_frontier now, hence no need to copy
  // back.
  CUDA_ERR_CHK(cudaMemcpy(visited, dev_visited, T, D2H));
  CUDA_ERR_CHK(cudaFree(dev_visited));
  CUDA_ERR_CHK(cudaFree(dev_frontier));

  for (uint32_t n = 0; n < num_vtx_; ++n) {
    if (visited[n]) {
      //      printf("\tvtx %lu is visited\n", n);
      cluster_ids[n] = cluster;
      if (memberships[n] != DBSCAN::membership::Core) {
        //        printf("\tmark %lu from Noise to Border\n", n);
        memberships[n] = DBSCAN::membership::Border;
      }
    }
  }

  delete[] visited;
  delete[] frontier;
}