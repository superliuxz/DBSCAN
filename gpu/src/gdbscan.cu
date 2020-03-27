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
  x_.resize(num_vtx_, 0);
  y_.resize(num_vtx_, 0);
  num_neighbours_.resize(num_vtx_, 0);
  start_pos_.resize(num_vtx_, 0);
  memberships.resize(num_vtx_, DBSCAN::membership::Noise);
  cluster_ids.resize(num_vtx_, -1);

  size_t n;
  float x, y;
  while (ifs >> n >> x >> y) {
    x_[n] = x;
    y_[n] = y;
  }
}

void GDBSCAN::Solver::calc_num_neighbours() {
  const auto num_blocks = std::ceil(num_vtx_ / static_cast<float>(BLOCK_SIZE));
  const auto N = sizeof(x_[0]) * num_vtx_;
  const auto K = sizeof(num_neighbours_[0]) * num_vtx_;

  printf("calc_num_neighbours needs: %lf MB\n",
         static_cast<double>(N + N + K) / 1024.f / 1024.f);

  // do not free dev_x_ and dev_y_; they are required to calculate
  // |neighbours_|
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_x_, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_y_, N));
  // do not free dev_num_neighbours_; it's required for the rest of algorithm.
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_num_neighbours_, K));
  CUDA_ERR_CHK(cudaMemcpy(dev_x_, thrust::raw_pointer_cast(x_.data()), N,
                          cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dev_y_, thrust::raw_pointer_cast(y_.data()), N,
                          cudaMemcpyHostToDevice));

  GDBSCAN::kernel_functions::k_num_nbs<<<num_blocks, BLOCK_SIZE>>>(
      dev_x_, dev_y_, dev_num_neighbours_, squared_radius_, num_vtx_);
  CUDA_ERR_CHK(cudaPeekAtLastError());

  CUDA_ERR_CHK(cudaMemcpy(thrust::raw_pointer_cast(num_neighbours_.data()),
                          dev_num_neighbours_, K, cudaMemcpyDeviceToHost));
}

void GDBSCAN::Solver::append_neighbours() {
  neighbours_.resize(start_pos_[num_vtx_ - 1] + num_neighbours_[num_vtx_ - 1],
                     0);
  //    printf("size of neighbours array: %lu\n", neighbours_.size());

  const auto num_blocks = std::ceil(num_vtx_ / static_cast<float>(BLOCK_SIZE));

  const auto N = sizeof(x_[0]) * num_vtx_;
  const auto K = sizeof(start_pos_[0]) * num_vtx_;
  const auto J = sizeof(neighbours_[0]) * neighbours_.size();

  printf("append_neighbours needs: %lf MB\n",
         static_cast<double>(N + N + K + J) / 1024.f / 1024.f);

  // |dev_x_| and |dev_y_| are in GPU memory.
  // Do not free |dev_start_pos_| and |dev_neighbours_|. They are required
  // for the rest of algorithm.
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_start_pos_, K));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_neighbours_, J));

  CUDA_ERR_CHK(cudaMemcpy(dev_start_pos_,
                          thrust::raw_pointer_cast(start_pos_.data()), K,
                          cudaMemcpyHostToDevice));

  GDBSCAN::kernel_functions::k_append_neighbours<<<num_blocks, BLOCK_SIZE>>>(
      dev_x_, dev_y_, dev_start_pos_, dev_neighbours_, num_vtx_,
      squared_radius_);
  CUDA_ERR_CHK(cudaPeekAtLastError());

  CUDA_ERR_CHK(cudaMemcpy(thrust::raw_pointer_cast(neighbours_.data()),
                          dev_neighbours_, J, cudaMemcpyDeviceToHost));
  // |dev_x_| and |dev_y_| are no longer used.
  CUDA_ERR_CHK(cudaFree(dev_x_));
  CUDA_ERR_CHK(cudaFree(dev_y_));
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
}

void GDBSCAN::Solver::bfs(const uint64_t u, const int cluster) {
  const auto num_blocks = std::ceil(num_vtx_ / static_cast<float>(BLOCK_SIZE));

  auto visited = new bool[num_vtx_]();
  auto border = new bool[num_vtx_]();
  uint64_t num_border = 1;
  border[u] = true;

  const auto T = sizeof(visited[0]) * num_vtx_;
  const auto N = sizeof(num_neighbours_[0]) * num_vtx_;
  const auto K = sizeof(neighbours_[0]) * neighbours_.size();
  const auto L = sizeof(DBSCAN::membership::Core) * num_vtx_;

  printf("bfs at %lu needs: %lf MB\n", u,
         static_cast<double>(T + T + N + N + K + L) / 1024.f / 1024.f);

  bool *dev_visited, *dev_border;
  DBSCAN::membership *dev_membership;
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_visited, T));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_border, T));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_membership, L))
  CUDA_ERR_CHK(cudaMemcpy(dev_visited, visited, T, cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dev_border, border, T, cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dev_membership,
                          thrust::raw_pointer_cast(memberships.data()), L,
                          cudaMemcpyHostToDevice));

  while (num_border > 0) {
    //    std::cout << "\t\tnum_border: " << num_border << std::endl;
    GDBSCAN::kernel_functions::k_bfs<<<num_blocks, BLOCK_SIZE>>>(
        dev_visited, dev_border, dev_num_neighbours_, dev_start_pos_,
        dev_neighbours_, dev_membership, num_vtx_);
    CUDA_ERR_CHK(cudaPeekAtLastError());
    num_border =
        thrust::count(thrust::device, dev_border, dev_border + num_vtx_, true);
  }
  // we don't care about he content in dev_border now, hence no need to copy
  // back.
  CUDA_ERR_CHK(cudaMemcpy(visited, dev_visited, T, cudaMemcpyDeviceToHost));
  CUDA_ERR_CHK(cudaFree(dev_visited));
  CUDA_ERR_CHK(cudaFree(dev_border));
  CUDA_ERR_CHK(cudaFree(dev_membership));

  for (uint64_t n = 0; n < num_vtx_; ++n) {
    if (visited[n]) {
      //      std::cout << "\tvtx " << n << " is visited" << std::endl;
      cluster_ids[n] = cluster;
      if (memberships[n] != DBSCAN::membership::Core)
        memberships[n] = DBSCAN::membership::Border;
    }
  }

  delete[] visited;
  delete[] border;
}