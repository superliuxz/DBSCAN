//
// Created by will on 2020-03-23.
//

#ifndef DBSCAN_GDBSCAN_CUH
#define DBSCAN_GDBSCAN_CUH

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "gdbscan_device_functions.cuh"
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

// TODO:
// - [x] construct Va and Ea, DO NOT use temp_adj. AVOID data transfer to GPU.
//   - [x] Va: thrust::exclusive_scan
//   - [x] Ea: one thread each node
// - [x] ID core and non-core: do this on CPU, which is faster than copying data
//     to GPU.
// - [x] BFS. no clue ... on thread each node? then ima hving a n^2 bfs???
// - Wrap all these non-kernel/non-device functions into Solver class
//
namespace GDBSCAN {
int const block_size = 512;

void calc_num_neighbours(const thrust::host_vector<float> &x,
                         const thrust::host_vector<float> &y,
                         thrust::host_vector<uint64_t> &num_nbs,
                         const float rad_sq) {
  assert(x.size() == y.size());
  assert(x.size() == num_nbs.size());

  const auto num_nodes = x.size();
  const auto num_blocks = std::ceil(num_nodes / static_cast<float>(block_size));
  const auto N = sizeof(x[0]) * num_nodes;
  const auto K = sizeof(num_nbs[0]) * num_nodes;

  printf("calc_num_neighbours needs: %lf MB\n",
         static_cast<double>(N + N + K) / 1024.f / 1024.f);

  float *dev_x, *dev_y;
  uint64_t *dev_num_nbs;
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_x, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_y, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_num_nbs, K));
  CUDA_ERR_CHK(cudaMemcpy(dev_x, thrust::raw_pointer_cast(x.data()), N,
                          cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dev_y, thrust::raw_pointer_cast(y.data()), N,
                          cudaMemcpyHostToDevice));

  GDBSCAN::kernel_functions::k_num_nbs<<<num_blocks, block_size>>>(
      dev_x, dev_y, dev_num_nbs, rad_sq, num_nodes);

  CUDA_ERR_CHK(cudaMemcpy(thrust::raw_pointer_cast(num_nbs.data()), dev_num_nbs,
                          K, cudaMemcpyDeviceToHost));
  CUDA_ERR_CHK(cudaFree(dev_x));
  CUDA_ERR_CHK(cudaFree(dev_y));
  CUDA_ERR_CHK(cudaFree(dev_num_nbs));
}

void calc_start_pos(const thrust::host_vector<uint64_t> &num_nbs,
                    thrust::host_vector<uint64_t> &start_pos) {
  // It's faster to computer the prefix sum on host because memcpy is avoided.
  thrust::exclusive_scan(thrust::host, num_nbs.cbegin(), num_nbs.cend(),
                         start_pos.begin());
}

void append_neighbours(const thrust::host_vector<float> &x,
                       const thrust::host_vector<float> &y,
                       const thrust::host_vector<uint64_t> &start_pos,
                       thrust::host_vector<uint64_t> &neighbours,
                       float const rad_sq) {
  assert(x.size() == y.size());
  assert(x.size() == start_pos.size());
  const auto num_nodes = x.size();
  const auto num_blocks = std::ceil(num_nodes / static_cast<float>(block_size));

  const auto N = sizeof(x[0]) * num_nodes;
  const auto K = sizeof(start_pos[0]) * num_nodes;
  const auto J = sizeof(neighbours[0]) * neighbours.size();

  printf("append_neighbours needs: %lf MB\n",
         static_cast<double>(N + N + K + J) / 1024.f / 1024.f);

  float *dev_x, *dev_y;
  uint64_t *dev_start_pos, *dev_neighbours;

  CUDA_ERR_CHK(cudaMalloc((void **)&dev_x, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_y, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_start_pos, K));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_neighbours, J));

  CUDA_ERR_CHK(cudaMemcpy(dev_x, thrust::raw_pointer_cast(x.data()), N,
                          cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dev_y, thrust::raw_pointer_cast(y.data()), N,
                          cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dev_start_pos,
                          thrust::raw_pointer_cast(start_pos.data()), K,
                          cudaMemcpyHostToDevice));

  GDBSCAN::kernel_functions::k_append_neighbours<<<num_blocks, block_size>>>(
      dev_x, dev_y, dev_start_pos, dev_neighbours, num_nodes, rad_sq);

  CUDA_ERR_CHK(cudaMemcpy(thrust::raw_pointer_cast(neighbours.data()),
                          dev_neighbours, J, cudaMemcpyDeviceToHost));
  CUDA_ERR_CHK(cudaFree(dev_x));
  CUDA_ERR_CHK(cudaFree(dev_y));
  CUDA_ERR_CHK(cudaFree(dev_start_pos));
  CUDA_ERR_CHK(cudaFree(dev_neighbours));
}

void bfs(const uint64_t u, const thrust::host_vector<uint64_t> &num_nbs,
         const thrust::host_vector<uint64_t> &start_pos,
         const thrust::host_vector<uint64_t> &neighbours,
         thrust::host_vector<DBSCAN::membership> &memberships,
         thrust::host_vector<int> &cluster_ids, const int cluster) {
  const auto num_nodes = num_nbs.size();
  const auto num_blocks = std::ceil(num_nodes / static_cast<float>(block_size));

  auto visited = new bool[num_nodes]();
  auto border = new bool[num_nodes]();
  uint64_t num_border = 1;
  border[u] = true;

  const auto T = sizeof(visited[0]) * num_nodes;
  const auto N = sizeof(num_nbs[0]) * num_nodes;
  const auto K = sizeof(neighbours[0]) * neighbours.size();
  const auto L = sizeof(DBSCAN::membership::Core) * num_nodes;

  printf("bfs at %lu needs: %lf MB\n", u,
         static_cast<double>(T + T + N + N + K + L) / 1024.f / 1024.f);

  bool *dev_visited, *dev_border;
  uint64_t *dev_num_nbs, *dev_start_pos, *dev_neighbours;
  DBSCAN::membership *dev_membership;
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_visited, T));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_border, T));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_num_nbs, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_start_pos, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_neighbours, K));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_membership, L))
  CUDA_ERR_CHK(cudaMemcpy(dev_visited, visited, T, cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dev_border, border, T, cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dev_num_nbs, thrust::raw_pointer_cast(num_nbs.data()),
                          N, cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dev_start_pos,
                          thrust::raw_pointer_cast(start_pos.data()), N,
                          cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dev_neighbours,
                          thrust::raw_pointer_cast(neighbours.data()), K,
                          cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dev_membership,
                          thrust::raw_pointer_cast(memberships.data()), L,
                          cudaMemcpyHostToDevice));

  while (num_border > 0) {
    //    std::cout << "\t\tnum_border: " << num_border << std::endl;
    GDBSCAN::kernel_functions::k_bfs<<<num_blocks, block_size>>>(
        dev_visited, dev_border, dev_num_nbs, dev_start_pos, dev_neighbours,
        dev_membership, num_nodes);
    num_border =
        thrust::count(thrust::device, dev_border, dev_border + num_nodes, true);
  }
  // we don't care about he content in dev_border now, hence no need to copy
  // back.
  CUDA_ERR_CHK(cudaMemcpy(visited, dev_visited, T, cudaMemcpyDeviceToHost));
  CUDA_ERR_CHK(cudaFree(dev_visited));
  CUDA_ERR_CHK(cudaFree(dev_border));
  CUDA_ERR_CHK(cudaFree(dev_num_nbs));
  CUDA_ERR_CHK(cudaFree(dev_start_pos));
  CUDA_ERR_CHK(cudaFree(dev_neighbours));
  CUDA_ERR_CHK(cudaFree(dev_membership));

  for (uint64_t n = 0; n < num_nodes; ++n) {
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

void identify_clusters(const thrust::host_vector<uint64_t> &num_nbs,
                       const thrust::host_vector<uint64_t> &start_pos,
                       const thrust::host_vector<uint64_t> &neighbours,
                       thrust::host_vector<DBSCAN::membership> &memberships,
                       thrust::host_vector<int> &cluster_ids) {
  assert(num_nbs.size() == start_pos.size());
  assert(neighbours.size() == start_pos.back() + num_nbs.back());
  const auto num_nodes = num_nbs.size();

  int cluster = 0;
  for (uint64_t u = 0; u < num_nodes; ++u) {
    if (cluster_ids[u] == -1 && memberships[u] == DBSCAN::membership::Core) {
      bfs(u, num_nbs, start_pos, neighbours, memberships, cluster_ids, cluster);
      ++cluster;
    }
  }
}
}  // namespace GDBSCAN

#endif  // DBSCAN_GDBSCAN_CUH