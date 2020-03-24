//
// Created by will on 2020-03-23.
//

#ifndef DBSCAN_GDBSCAN_CUH
#define DBSCAN_GDBSCAN_CUH

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

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
// - ID core and non-core: one thread each node
// - BFS. no clue ... on thread each node? then ima hving a n^2 bfs???
//
namespace GDBSCAN {
int const blocksize = 512;

__device__ float square_dist(const float &x1, const float &y1, const float &x2,
                             const float &y2) {
  return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

// Kernel to calculate the number of neighbours of each node
__global__ void k_num_nbs(float const *const x, float const *const y,
                          uint64_t *const num_nbs, const float rad_sq,
                          const uint64_t num_nodes) {
  uint64_t const u = threadIdx.x + blockIdx.x * blockDim.x;
  if (u >= num_nodes) return;
  num_nbs[u] = 0;
  for (auto v = 0u; v < num_nodes; ++v) {
    if (u != v && square_dist(x[u], y[u], x[v], y[v]) <= rad_sq) ++num_nbs[u];
  }
}

void calc_num_neighbours(float const *const x, float const *const y,
                         uint64_t *const num_nbs, const float &rad_sq,
                         const uint64_t &num_nodes) {
  const auto num_blocks = std::ceil(num_nodes / static_cast<float>(blocksize));

  const auto N = sizeof(x[0]) * num_nodes;
  const auto K = sizeof(num_nbs[0]) * num_nodes;

  printf("calc_num_neighbours needs: %lf MB\n",
         static_cast<double>(N + N + K) / 1024.f / 1024.f);

  float *dev_x, *dev_y;
  uint64_t *dev_num_nbs;
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_x, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_y, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_num_nbs, K));
  CUDA_ERR_CHK(cudaMemcpy(dev_x, x, N, cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dev_y, y, N, cudaMemcpyHostToDevice));

  k_num_nbs<<<num_blocks, blocksize>>>(dev_x, dev_y, dev_num_nbs, rad_sq,
                                       num_nodes);

  CUDA_ERR_CHK(cudaMemcpy(num_nbs, dev_num_nbs, K, cudaMemcpyDeviceToHost));
  CUDA_ERR_CHK(cudaFree(dev_x));
  CUDA_ERR_CHK(cudaFree(dev_y));
  CUDA_ERR_CHK(cudaFree(dev_num_nbs));
}

void calc_start_pos(uint64_t const *const num_nbs, uint64_t *const start_pos,
                    const uint64_t num_nodes) {
  thrust::exclusive_scan(thrust::host, num_nbs, num_nbs + num_nodes, start_pos);
}

// Kernel to populate the actual neighbours array
__global__ void k_append_neighbours(float const *const x, float const *const y,
                                    uint64_t const *const start_pos,
                                    uint64_t *const neighbours,
                                    const uint64_t num_nodes,
                                    const float rad_sq) {
  uint64_t const u = threadIdx.x + blockIdx.x * blockDim.x;
  if (u >= num_nodes) return;
  uint64_t upos = start_pos[u];
  for (uint64_t v = 0u; v < num_nodes; ++v) {
    if (u != v && square_dist(x[u], y[u], x[v], y[v]) <= rad_sq) {
      neighbours[upos++] = v;
    }
  }
}

void append_neighbours(float const *const x, float const *const y,
                       uint64_t const *const start_pos,
                       uint64_t *const neighbours, const uint64_t num_nodes,
                       uint64_t const nb_arr_sz, float const rad_sq) {
  const auto num_blocks = std::ceil(num_nodes / static_cast<float>(blocksize));

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

  k_append_neighbours<<<num_blocks, blocksize>>>(
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