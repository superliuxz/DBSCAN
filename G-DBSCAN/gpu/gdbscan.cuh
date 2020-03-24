//
// Created by will on 2020-03-23.
//

#ifndef DBSCAN_GDBSCAN_CUH
#define DBSCAN_GDBSCAN_CUH

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

// TODO:
// - construct Va and Ea, DO NOT use temp_adj. AVOID data transfer to GPU.
//   - [x] Va: thrust::exclusive_scan
//   - Ea: one thread each node
// - ID core and non-core: one thread each node
// - BFS. no clue ... on thread each node? then ima hving a n^2 bfs???
//
namespace GDBSCAN {
int const blocksize = 512;

__device__ float square_dist(const float &x1, const float &y1, const float &x2,
                             const float &y2) {
  return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

// Kernel to populate Va
__global__ void pop_va(float const *const x, float const *const y,
                       uint64_t *const Va, const float rad_sq,
                       const uint64_t num_nodes) {
  uint64_t const u = threadIdx.x + blockIdx.x * blockDim.x;
  if (u >= num_nodes) return;
  uint64_t num_nbs = 0;
  for (auto v = 0u; v < num_nodes; ++v) {
    if (u != v && square_dist(x[u], y[u], x[v], y[v]) <= rad_sq) ++num_nbs;
  }
  Va[u] = num_nbs;
}

void insert_edge(float const *const x, float const *const y, uint64_t *const Va,
                 const float &rad_sq, const uint64_t &num_nodes) {
  const auto coord_size = sizeof(x[0]) * num_nodes;
  const auto Va_size = sizeof(Va[0]) * num_nodes;

  const auto num_blocks = std::ceil(num_nodes / static_cast<float>(blocksize));
  float *dev_x, *dev_y;
  uint64_t *dev_Va;
  cudaMalloc((void **)&dev_x, coord_size);
  cudaMalloc((void **)&dev_y, coord_size);
  cudaMalloc((void **)&dev_Va, Va_size);
  cudaMemcpy(dev_x, x, coord_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y, y, coord_size, cudaMemcpyHostToDevice);

  pop_va<<<num_blocks, blocksize>>>(dev_x, dev_y, dev_Va, rad_sq, num_nodes);

  cudaDeviceSynchronize();

  cudaMemcpy(Va, dev_Va, Va_size, cudaMemcpyDeviceToHost);
  cudaFree(dev_x);
  cudaFree(dev_y);
  cudaFree(dev_Va);
}

void calc_Va(uint64_t *const Va, const uint64_t num_nodes) {
  thrust::exclusive_scan(thrust::host, Va, Va + num_nodes, Va);
}

}  // namespace GDBSCAN

#endif  // DBSCAN_GDBSCAN_CUH