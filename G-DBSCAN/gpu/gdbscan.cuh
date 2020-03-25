//
// Created by will on 2020-03-23.
//

#ifndef DBSCAN_GDBSCAN_CUH
#define DBSCAN_GDBSCAN_CUH

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "gdbscan_device_functions.cuh"
#include "gdbscan_kernel_functions.cuh"

#define CUDA_ERR_CHK(code) \
  { cuda_err_chk((code), __FILE__, __LINE__); }
inline void cuda_err_chk(cudaError_t code, const char *file, int line,
                         bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA_ERR_CHK: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

// TODO:
// - construct Va and Ea, DO NOT use temp_adj. AVOID data transfer to GPU.
//   - [x] Va: thrust::exclusive_scan
//   - [x] Ea: one thread each node
// - ID core and non-core: do this on CPU, which is faster than copying data
//     to GPU.
// - BFS. no clue ... on thread each node? then ima hving a n^2 bfs???
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

void calc_start_pos(uint64_t const *const num_nbs, uint64_t *const start_pos,
                    const uint64_t num_nodes) {
  thrust::exclusive_scan(thrust::host, num_nbs, num_nbs + num_nodes, start_pos);
}

void append_neighbours(float const *const x, float const *const y,
                       uint64_t const *const start_pos,
                       uint64_t *const neighbours, const uint64_t num_nodes,
                       uint64_t const nb_arr_sz, float const rad_sq) {
  const auto num_blocks = std::ceil(num_nodes / static_cast<float>(block_size));

  const auto N = sizeof(x[0]) * num_nodes;
  const auto K = sizeof(start_pos[0]) * num_nodes;
  const auto J = sizeof(neighbours[0]) * nb_arr_sz;

  printf("append_neighbours needs: %lf MB\n",
         static_cast<double>(N + N + K + J) / 1024.f / 1024.f);

  float *dev_x, *dev_y;
  uint64_t *dev_start_pos, *dev_neighbours;

  CUDA_ERR_CHK(cudaMalloc((void **)&dev_x, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_y, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_start_pos, K));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_neighbours, J));

  CUDA_ERR_CHK(cudaMemcpy(dev_x, x, N, cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dev_y, y, N, cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dev_start_pos, start_pos, K, cudaMemcpyHostToDevice));

  GDBSCAN::kernel_functions::k_append_neighbours<<<num_blocks, block_size>>>(
      dev_x, dev_y, dev_start_pos, dev_neighbours, num_nodes, rad_sq);

  CUDA_ERR_CHK(
      cudaMemcpy(neighbours, dev_neighbours, J, cudaMemcpyDeviceToHost));
  CUDA_ERR_CHK(cudaFree(dev_x));
  CUDA_ERR_CHK(cudaFree(dev_y));
  CUDA_ERR_CHK(cudaFree(dev_start_pos));
  CUDA_ERR_CHK(cudaFree(dev_neighbours));
}
}  // namespace GDBSCAN

#endif  // DBSCAN_GDBSCAN_CUH