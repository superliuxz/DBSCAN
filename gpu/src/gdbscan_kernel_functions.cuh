//
// Created by will on 2020-03-24.
//

#ifndef DBSCAN_GDBSCAN_KERNEL_FUNCTIONS_CUH
#define DBSCAN_GDBSCAN_KERNEL_FUNCTIONS_CUH

#include <DBSCAN/membership.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <cmath>

#include "gdbscan_device_functions.cuh"

namespace GDBSCAN {
namespace kernel_functions {
constexpr uint32_t kSharedMemBytes = 24 * 1024;

/*!
 * Calculate the number of neighbours of each vertex. One kernel thread per
 * vertex.
 * @param x - x values, sorted by l1 norm.
 * @param y - y values, sorted by l1 norm.
 * @param l1norm - sorted l1 norm.
 * @param vtx_mapper - maps sorted vertex index to original.
 * @param rad - radius.
 * @param num_vtx - number of vertices.
 * @param num_nbs - output array.
 */
__global__ void k_num_nbs(float const *const x, float const *const y,
                          float const *const l1norm,
                          uint32_t const *const vtx_mapper, const float rad,
                          const uint32_t num_vtx, uint32_t *const num_nbs) {
  uint32_t const u = threadIdx.x + blockIdx.x * blockDim.x;
  if (u >= num_vtx) return;
  // first vtx of current block.
  const uint32_t tb_start = blockIdx.x * blockDim.x;
  // last vtx of current block.
  const uint32_t tb_end = min(tb_start + blockDim.x, num_vtx) - 1;

  int land_id = threadIdx.x & 0x1f;
  float const *possible_range_start, *possible_range_end;
  if (land_id == 0) {
    // inclusive start
    possible_range_start = thrust::lower_bound(
        thrust::device, l1norm, l1norm + num_vtx, l1norm[tb_start] - 2 * rad);
    // exclusive end
    possible_range_end = thrust::upper_bound(
        thrust::device, l1norm, l1norm + num_vtx, l1norm[tb_end] + 2 * rad);
  }
  possible_range_start =
      (float *)__shfl_sync(0xffffffff, (uint64_t)possible_range_start, 0);
  possible_range_end =
      (float *)__shfl_sync(0xffffffff, (uint64_t)possible_range_end, 0);

  // the number of threads might not be blockDim.x, if this is the last block.
  uint32_t const num_threads = tb_end - tb_start + 1;
  const uint32_t tile_size = kSharedMemBytes / 4 / (1 + 1);
  // first half of shared stores Xs; second half stores Ys.
  __shared__ float shared[tile_size * (1 + 1)];
  auto *const sh_x = shared;
  auto *const sh_y = shared + tile_size;
  uint32_t ans = 0;

  for (auto curr_ptr = possible_range_start; curr_ptr < possible_range_end;
       curr_ptr += tile_size) {
    // curr_ptr's index
    uint32_t const curr_idx = curr_ptr - l1norm;
    // current range; might be less than tile_size.
    uint32_t const curr_range =
        min(tile_size, static_cast<uint32_t>(possible_range_end - curr_ptr));
    // thread 0 updates sh_x[0], sh_x[0+num_threads], sh_x[0+2*num_threads] ...
    // thread 1 updates sh_x[1], sh_x[1+num_threads], sh_x[1+2*num_threads] ...
    // ...
    // thread t updates sh_x[t], sh_x[t+num_threads], sh_x[t+2*num_threads] ...
    __syncthreads();
    for (auto i = threadIdx.x; i < curr_range; i += num_threads) {
      sh_x[i] = x[curr_idx + i];
      sh_y[i] = y[curr_idx + i];
    }
    __syncthreads();
    const float ux = x[u], uy = y[u];
    for (auto j = 0; j < curr_range; ++j) {
      ans += GDBSCAN::device_functions::square_dist(ux, uy, sh_x[j], sh_y[j]) <=
             rad * rad;
    }
  }
  num_nbs[vtx_mapper[u]] = ans - 1;
}
/*!
 * Populate the neighbours array. One kernel thread per vertex.
 * @param x - x values, sorted by l1 norm.
 * @param y - y values, sorted by l1 norm.
 * @param l1norm - sorted l1 norm.
 * @param vtx_mapper - maps sorted vertex index to original.
 * @param start_pos - neighbours starting index of each vertex.
 * @param rad - radius.
 * @param num_vtx - number of vertices
 * @param neighbours - output array
 */
__global__ void k_append_neighbours(float const *const x, float const *const y,
                                    float const *const l1norm,
                                    uint32_t const *const vtx_mapper,
                                    uint32_t const *const start_pos,
                                    const float rad, const uint32_t num_vtx,
                                    uint32_t *const neighbours) {
  uint32_t const u = threadIdx.x + blockIdx.x * blockDim.x;
  if (u >= num_vtx) return;
  // first vtx of current block.
  const uint32_t tb_start = blockIdx.x * blockDim.x;
  // last vtx of current block.
  const uint32_t tb_end = min(tb_start + blockDim.x, num_vtx) - 1;

  int land_id = threadIdx.x & 0x1f;
  float const *possible_range_start, *possible_range_end;
  if (land_id == 0) {
    // inclusive start
    possible_range_start = thrust::lower_bound(
        thrust::device, l1norm, l1norm + num_vtx, l1norm[tb_start] - 2 * rad);
    // exclusive end
    possible_range_end = thrust::upper_bound(
        thrust::device, l1norm, l1norm + num_vtx, l1norm[tb_end] + 2 * rad);
  }
  possible_range_start =
      (float *)__shfl_sync(0xffffffff, (uint64_t)possible_range_start, 0);
  possible_range_end =
      (float *)__shfl_sync(0xffffffff, (uint64_t)possible_range_end, 0);

  uint32_t const num_threads = tb_end - tb_start + 1;
  // different from previous kernel, here the shared array is tri-partitioned,
  // because of the frequent access to vtx_mapper.
  const uint32_t tile_size = kSharedMemBytes / 4 / (1 + 1 + 1);
  __shared__ float shared[tile_size * (1 + 1 + 1)];
  auto *const sh_x = shared;
  auto *const sh_y = shared + tile_size;
  auto *const sh_vtx_mapper = (uint32_t *)(sh_y + tile_size);
  uint32_t upos = start_pos[vtx_mapper[u]];

  for (auto curr_ptr = possible_range_start; curr_ptr < possible_range_end;
       curr_ptr += tile_size) {
    // curr_ptr's index
    uint32_t const curr_idx = curr_ptr - l1norm;
    // current range; might be less than tile_size.
    uint32_t const curr_range =
        min(tile_size, static_cast<uint32_t>(possible_range_end - curr_ptr));
    // thread 0 updates sh_x[0], sh_x[0+num_threads], sh_x[0+2*num_threads] ...
    // thread 1 updates sh_x[1], sh_x[1+num_threads], sh_x[1+2*num_threads] ...
    // ...
    // thread t updates sh_x[t], sh_x[t+num_threads], sh_x[t+2*num_threads] ...
    __syncthreads();
    for (auto i = threadIdx.x; i < curr_range; i += num_threads) {
      sh_x[i] = x[curr_idx + i];
      sh_y[i] = y[curr_idx + i];
      sh_vtx_mapper[i] = vtx_mapper[curr_idx + i];
    }
    __syncthreads();
    const float ux = x[u], uy = y[u];
    for (auto j = 0; j < curr_range; ++j) {
      // the if guard is faster than a branchless because the write to
      // neighbours is slow. In the branchless version, neighbours is updated
      // for every j.
      if (u != curr_idx + j && GDBSCAN::device_functions::square_dist(
                                   ux, uy, sh_x[j], sh_y[j]) <= rad * rad) {
        neighbours[upos++] = sh_vtx_mapper[j];
      }
    }
  }
}
/*!
 * Identify all the Core vertices. One kernel thread per vertex.
 * @param num_neighbours - the number of neighbours of each vertex.
 * @param membership - membership of each vertex.
 * @param num_vtx - number of vertex.
 * @param min_pts - query parameter, minimum number of points to be consider as
 * a Core.
 */
__global__ void k_identify_cores(uint32_t const *const num_neighbours,
                                 DBSCAN::membership *const membership,
                                 const uint32_t num_vtx,
                                 const uint32_t min_pts) {
  uint32_t const u = threadIdx.x + blockIdx.x * blockDim.x;
  if (u >= num_vtx) return;
  membership[u] = DBSCAN::membership::Noise;
  if (num_neighbours[u] >= min_pts) membership[u] = DBSCAN::membership::Core;
}
/*!
 * Traverse the graph from each vertex. One kernel thread per vertex.
 * @param visited - boolean array that tracks if a vertex has been visited.
 * @param frontier - boolean array that tracks if a vertex is on the frontier.
 * @param num_nbs - the number of neighbours of each vertex.
 * @param start_pos - neighbours starting index of each vertex.
 * @param neighbours - the actually neighbour indices of each vertex.
 * @param membership - membership of each vertex.
 * @param num_vtx - number of vertices of the graph.
 */
__global__ void k_bfs(bool *const visited, bool *const frontier,
                      uint32_t const *const num_nbs,
                      uint32_t const *const start_pos,
                      uint32_t const *const neighbours,
                      DBSCAN::membership const *const membership,
                      uint32_t num_vtx) {
  uint32_t const u = threadIdx.x + blockIdx.x * blockDim.x;
  if (u >= num_vtx) return;
  if (!frontier[u]) return;
  //  printf("\t\ttmark %lu visited and remove from frontier\n", u);
  frontier[u] = false;
  visited[u] = true;
  // Stop BFS if u is not Core.
  if (membership[u] != DBSCAN::membership::Core) return;
  uint32_t u_start = start_pos[u];
  //  printf("\t\t%lu has %lu nbs starting from idx %lu\n", u, num_nbs[u],
  //  u_start);
  for (uint32_t i = 0; i < num_nbs[u]; ++i) {
    uint32_t v = neighbours[u_start + i];
    // scarifies divergence for less global memory access.
    // `frontier[v] = !visited[v];` would make it branch-less at the cost of
    // more global write.
    if (!visited[v]) frontier[v] = true;
  }
}
}  // namespace kernel_functions
}  // namespace GDBSCAN
#endif  // DBSCAN_GDBSCAN_KERNEL_FUNCTIONS_CUH
