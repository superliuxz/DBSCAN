//
// Created by will on 2020-03-23.
//

#ifndef DBSCAN_GDBSCAN_CUH
#define DBSCAN_GDBSCAN_CUH

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <chrono>

#include "gdbscan_device_functions.cuh"
#include "gdbscan_kernel_functions.cuh"
#include "membership.h"
#include "utils.h"

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

namespace GDBSCAN {
int const BLOCK_SIZE = 512;

class Solver {
 public:
  Solver(const std::string &input, const uint64_t &min_pts, const float &radius)
      : squared_radius_(radius * radius), min_pts_(min_pts) {
    using namespace std::chrono;
    high_resolution_clock::time_point start = high_resolution_clock::now();

    auto ifs = std::ifstream(input);
    ifs >> num_vtx_;
    x_.resize(num_vtx_, 0);
    y_.resize(num_vtx_, 0);
    num_neighbours_.resize(num_vtx_, 0);
    start_pos_.resize(num_vtx_, 0);
    membership_.resize(num_vtx_, DBSCAN::membership::Noise);
    cluster_ids_.resize(num_vtx_, -1);

    size_t n;
    float x, y;
    while (ifs >> n >> x >> y) {
      x_[n] = x;
      y_[n] = y;
    }

    duration<double> time_spent =
        duration_cast<duration<double>>(high_resolution_clock::now() - start);
    printf("reading vertices takes %lf seconds\n", time_spent.count());
  }

  void calc_num_neighbours() {
    assert(x_.size() == y_.size());
    assert(x_.size() == num_neighbours_.size());

    const auto num_nodes = x_.size();
    const auto num_blocks =
        std::ceil(num_nodes / static_cast<float>(BLOCK_SIZE));
    const auto N = sizeof(x_[0]) * num_nodes;
    const auto K = sizeof(num_neighbours_[0]) * num_nodes;

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
        dev_x_, dev_y_, dev_num_neighbours_, squared_radius_, num_nodes);

    CUDA_ERR_CHK(cudaMemcpy(thrust::raw_pointer_cast(num_neighbours_.data()),
                            dev_num_neighbours_, K, cudaMemcpyDeviceToHost));
  }

  void calc_start_pos() {
    // It's faster to computer the prefix sum on host because memcpy is avoided.
    thrust::exclusive_scan(thrust::host, num_neighbours_.cbegin(),
                           num_neighbours_.cend(), start_pos_.begin());
  }

  void append_neighbours() {
    neighbours_.resize(start_pos_[num_vtx_ - 1] + num_neighbours_[num_vtx_ - 1],
                       0);
    printf("size of neighbours array: %lu\n", neighbours_.size());

    const auto num_nodes = x_.size();
    const auto num_blocks =
        std::ceil(num_nodes / static_cast<float>(BLOCK_SIZE));

    const auto N = sizeof(x_[0]) * num_nodes;
    const auto K = sizeof(start_pos_[0]) * num_nodes;
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
        dev_x_, dev_y_, dev_start_pos_, dev_neighbours_, num_nodes,
        squared_radius_);

    CUDA_ERR_CHK(cudaMemcpy(thrust::raw_pointer_cast(neighbours_.data()),
                            dev_neighbours_, J, cudaMemcpyDeviceToHost));
    // |dev_x_| and |dev_y_| are no longer used.
    CUDA_ERR_CHK(cudaFree(dev_x_));
    CUDA_ERR_CHK(cudaFree(dev_y_));
  }

  void identify_cores() {
    for (uint64_t i = 0; i < num_vtx_; ++i) {
      if (num_neighbours_[i] >= min_pts_)
        membership_[i] = DBSCAN::membership::Core;
    }
  }

  void identify_clusters() {
    const auto num_nodes = num_neighbours_.size();

    int cluster = 0;
    for (uint64_t u = 0; u < num_nodes; ++u) {
      if (cluster_ids_[u] == -1 && membership_[u] == DBSCAN::membership::Core) {
        bfs(u, cluster);
        ++cluster;
      }
    }
    CUDA_ERR_CHK(cudaFree(dev_num_neighbours_));
    CUDA_ERR_CHK(cudaFree(dev_start_pos_));
    CUDA_ERR_CHK(cudaFree(dev_neighbours_));
  }

#if !defined(TESTING)
 public:
#else
 private:
#endif
  // query params
  float squared_radius_;
  uint64_t num_vtx_{};
  uint64_t min_pts_;
  // data structures
  std::vector<float> x_, y_;
  std::vector<uint64_t> num_neighbours_, start_pos_;
  std::vector<uint64_t, DBSCAN::utils::NonConstructAllocator<uint64_t>>
      neighbours_;
  std::vector<DBSCAN::membership> membership_;
  std::vector<int> cluster_ids_;
  // gpu vars. Class members to avoid unnecessary copy.
  float *dev_x_{}, *dev_y_{};
  uint64_t *dev_num_neighbours_{}, *dev_start_pos_{}, *dev_neighbours_{};

 private:
  void bfs(const uint64_t u, const int cluster) {
    const auto num_nodes = num_neighbours_.size();
    const auto num_blocks =
        std::ceil(num_nodes / static_cast<float>(BLOCK_SIZE));

    auto visited = new bool[num_nodes]();
    auto border = new bool[num_nodes]();
    uint64_t num_border = 1;
    border[u] = true;

    const auto T = sizeof(visited[0]) * num_nodes;
    const auto N = sizeof(num_neighbours_[0]) * num_nodes;
    const auto K = sizeof(neighbours_[0]) * neighbours_.size();
    const auto L = sizeof(DBSCAN::membership::Core) * num_nodes;

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
                            thrust::raw_pointer_cast(membership_.data()), L,
                            cudaMemcpyHostToDevice));

    while (num_border > 0) {
      //    std::cout << "\t\tnum_border: " << num_border << std::endl;
      GDBSCAN::kernel_functions::k_bfs<<<num_blocks, BLOCK_SIZE>>>(
          dev_visited, dev_border, dev_num_neighbours_, dev_start_pos_,
          dev_neighbours_, dev_membership, num_nodes);
      num_border = thrust::count(thrust::device, dev_border,
                                 dev_border + num_nodes, true);
    }
    // we don't care about he content in dev_border now, hence no need to copy
    // back.
    CUDA_ERR_CHK(cudaMemcpy(visited, dev_visited, T, cudaMemcpyDeviceToHost));
    CUDA_ERR_CHK(cudaFree(dev_visited));
    CUDA_ERR_CHK(cudaFree(dev_border));
    CUDA_ERR_CHK(cudaFree(dev_membership));

    for (uint64_t n = 0; n < num_nodes; ++n) {
      if (visited[n]) {
        //      std::cout << "\tvtx " << n << " is visited" << std::endl;
        cluster_ids_[n] = cluster;
        if (membership_[n] != DBSCAN::membership::Core)
          membership_[n] = DBSCAN::membership::Border;
      }
    }

    delete[] visited;
    delete[] border;
  }
};
}  // namespace GDBSCAN

#endif  // DBSCAN_GDBSCAN_CUH