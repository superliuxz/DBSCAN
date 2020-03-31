//
// Created by will on 2020-03-26.
//

#include <thrust/count.h>

#include <fstream>

#include "gdbscan.cuh"
#include "gdbscan_kernel_functions.cuh"
#include "membership.h"

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

GDBSCAN::Solver::Solver(const std::string &input, const uint64_t &min_pts,
                        const float &radius)
    : squared_radius_(radius * radius), min_pts_(min_pts) {
  auto ifs = std::ifstream(input);
  ifs >> num_vtx_;
  num_blocks = std::ceil(num_vtx_ / static_cast<float>(BLOCK_SIZE));
  x_.resize(num_vtx_, 0);
  y_.resize(num_vtx_, 0);
  memberships.resize(num_vtx_, DBSCAN::membership::Noise);
  cluster_ids.resize(num_vtx_, -1);

  size_t n;
  float x, y;
  while (ifs >> n >> x >> y) {
    x_[n] = x;
    y_[n] = y;
  }
#if GDBSCAN_TESTING == 1
  start_pos.resize(num_vtx_, 0);
  num_neighbours.resize(num_vtx_, 0);
#endif
}

void GDBSCAN::Solver::calc_num_neighbours() {
  const auto N = sizeof(x_[0]) * num_vtx_;
  const auto K = sizeof(*dev_num_neighbours_) * num_vtx_;

  printf("calc_num_neighbours needs: %lf MB\n",
         static_cast<double>(N + N + K) / 1024.f / 1024.f);

  uint64_t last_vtx_num_nbs = 0;
  // do not free dev_x_ and dev_y_; they are required to calculate
  // |neighbours|
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_x_, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_y_, N));
  // do not free |dev_num_neighbours_|; it's required for the rest of algorithm.
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_num_neighbours_, K));
  CUDA_ERR_CHK(cudaMemcpy(dev_x_, thrust::raw_pointer_cast(x_.data()), N, H2D));
  CUDA_ERR_CHK(cudaMemcpy(dev_y_, thrust::raw_pointer_cast(y_.data()), N, H2D));

  GDBSCAN::kernel_functions::k_num_nbs<<<num_blocks, BLOCK_SIZE>>>(
      dev_x_, dev_y_, dev_num_neighbours_, squared_radius_, num_vtx_);
  CUDA_ERR_CHK(cudaPeekAtLastError());
  CUDA_ERR_CHK(cudaMemcpy(&last_vtx_num_nbs, dev_num_neighbours_ + num_vtx_ - 1,
                          sizeof(last_vtx_num_nbs), D2H));

#if GDBSCAN_TESTING == 1
  CUDA_ERR_CHK(cudaMemcpy(thrust::raw_pointer_cast(num_neighbours.data()),
                          dev_num_neighbours_, K, D2H));
#endif
  total_num_nbs_ += last_vtx_num_nbs;
  // |dev_x_|, |dev_y_|, |dev_num_neighbours_| in GPU RAM.
}

void GDBSCAN::Solver::calc_start_pos() {
  uint64_t last_vtx_start_pos = 0;

  const auto N = sizeof(dev_start_pos_[0]) * num_vtx_;

  printf("calc_start_pos needs: %lf MB\n",
         static_cast<double>(N) / 1024.f / 1024.f);
  // Do not free |dev_start_pos_|. It's required for the rest of algorithm.
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_start_pos_, N));
  thrust::exclusive_scan(thrust::device, dev_num_neighbours_,
                         dev_num_neighbours_ + num_vtx_, dev_start_pos_);
  CUDA_ERR_CHK(cudaMemcpy(&last_vtx_start_pos, dev_start_pos_ + num_vtx_ - 1,
                          sizeof(uint64_t), D2H));
#if GDBSCAN_TESTING == 1
  CUDA_ERR_CHK(cudaMemcpy(thrust::raw_pointer_cast(start_pos.data()),
                          dev_start_pos_, N, D2H));
#endif
  total_num_nbs_ += last_vtx_start_pos;
  // |dev_x_|, |dev_y_|, |dev_num_neighbours_|, |dev_start_pos_| in GPU RAM.
}

void GDBSCAN::Solver::append_neighbours() {
#if GDBSCAN_TESTING == 1
  neighbours.resize(total_num_nbs_, 0);
#endif
  //  printf("size of neighbours array: %lu\n", total_num_nbs_);

  const auto N = sizeof(x_[0]) * num_vtx_;
  const auto K = sizeof(*dev_start_pos_) * num_vtx_;
  const auto J = sizeof(*dev_neighbours_) * total_num_nbs_;

  printf("append_neighbours needs: %lf MB\n",
         static_cast<double>(N + N + K + J) / 1024.f / 1024.f);

  // |dev_x_| and |dev_y_| are in GPU memory.
  // Do not free |dev_neighbours_|. It's required for the rest of algorithm.
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_neighbours_, J));

  GDBSCAN::kernel_functions::k_append_neighbours<<<num_blocks, BLOCK_SIZE>>>(
      dev_x_, dev_y_, dev_start_pos_, dev_neighbours_, num_vtx_,
      squared_radius_);
  CUDA_ERR_CHK(cudaPeekAtLastError());

  // |dev_x_| and |dev_y_| are no longer used.
  CUDA_ERR_CHK(cudaFree(dev_x_));
  CUDA_ERR_CHK(cudaFree(dev_y_));
#if GDBSCAN_TESTING == 1
  CUDA_ERR_CHK(cudaMemcpy(thrust::raw_pointer_cast(neighbours.data()),
                          dev_neighbours_, J, D2H));
#endif
  // |dev_num_neighbours_|, |dev_start_pos_|, |dev_neighbours_| in GPU RAM.
}

void GDBSCAN::Solver::identify_cores() {
  const auto N = sizeof(*dev_num_neighbours_) * num_vtx_;
  const auto M = sizeof(*dev_membership_) * num_vtx_;
  printf("identify_cores needs: %lf MB\n",
         static_cast<double>(M + N) / 1024.f / 1024.f);
  // Do not free |dev_membership_| as it's used in BFS. The content will be
  // copied out at the end of |identify_clusters|.
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_membership_, M));
  GDBSCAN::kernel_functions::k_identify_cores<<<num_blocks, BLOCK_SIZE>>>(
      dev_num_neighbours_, dev_membership_, num_vtx_, min_pts_);
  CUDA_ERR_CHK(cudaMemcpy(thrust::raw_pointer_cast(memberships.data()),
                          dev_membership_, M, D2H));
  // |dev_num_neighbours_|, |dev_start_pos_|, |dev_neighbours_|,
  // |dev_membership_| in GPU RAM.
}

void GDBSCAN::Solver::identify_clusters() {
  int cluster = 0;
  for (uint64_t u = 0; u < num_vtx_; ++u) {
    if (cluster_ids[u] == -1 && memberships[u] == DBSCAN::membership::Core) {
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

void GDBSCAN::Solver::bfs(const uint64_t u, const int cluster) {
  auto visited = new bool[num_vtx_]();
  auto border = new bool[num_vtx_]();
  uint64_t num_border = 1;
  border[u] = true;

  const auto T = sizeof(visited[0]) * num_vtx_;
  const auto N = sizeof(*dev_num_neighbours_) * num_vtx_;
  const auto L = sizeof(*dev_membership_) * num_vtx_;
  const auto K = sizeof(*dev_neighbours_) * total_num_nbs_;

  printf("bfs at %lu needs: %lf MB\n", u,
         static_cast<double>(T + T + N + N + K + L) / 1024.f / 1024.f);

  bool *dev_visited, *dev_border;
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_visited, T));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_border, T));
  CUDA_ERR_CHK(cudaMemcpy(dev_visited, visited, T, H2D));
  CUDA_ERR_CHK(cudaMemcpy(dev_border, border, T, H2D));
  CUDA_ERR_CHK(cudaMemcpy(
      dev_membership_, thrust::raw_pointer_cast(memberships.data()), L, H2D));

  while (num_border > 0) {
    //    printf("\tnumber_border: %lu\n", num_border);
    GDBSCAN::kernel_functions::k_bfs<<<num_blocks, BLOCK_SIZE>>>(
        dev_visited, dev_border, dev_num_neighbours_, dev_start_pos_,
        dev_neighbours_, dev_membership_, num_vtx_);
    CUDA_ERR_CHK(cudaPeekAtLastError());
    num_border =
        thrust::count(thrust::device, dev_border, dev_border + num_vtx_, true);
  }
  // we don't care about he content in dev_border now, hence no need to copy
  // back.
  CUDA_ERR_CHK(cudaMemcpy(visited, dev_visited, T, D2H));
  CUDA_ERR_CHK(cudaFree(dev_visited));
  CUDA_ERR_CHK(cudaFree(dev_border));

  for (uint64_t n = 0; n < num_vtx_; ++n) {
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
  delete[] border;
}