//
// Created by will on 2020-03-24.
//

#ifndef DBSCAN_GDBSCAN_KERNEL_FUNCTIONS_CUH
#define DBSCAN_GDBSCAN_KERNEL_FUNCTIONS_CUH

#include <membership.h>
#include <thrust/binary_search.h>

#include <cmath>

#include "gdbscan_device_functions.cuh"

namespace GDBSCAN {
namespace kernel_functions {
/*!
 * Calculate the number of neighbours of each vertex. One kernel per vertex.
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
                          uint64_t const *const vtx_mapper, const float rad,
                          const uint64_t num_vtx, uint64_t *const num_nbs) {
  uint64_t const u = threadIdx.x + blockIdx.x * blockDim.x;
  if (u >= num_vtx) return;
  // first vtx of current block.
  const uint64_t tb_start = blockIdx.x * blockDim.x;
  // last vtx of current block.
  const uint64_t tb_end = min(tb_start + blockDim.x, num_vtx - 1);
  // inclusive start
  const float *range_start = thrust::lower_bound(
      thrust::device, l1norm, l1norm + num_vtx, l1norm[tb_start] - 2 * rad);
  // exclusive end
  const float *range_end = thrust::upper_bound(
      thrust::device, l1norm, l1norm + num_vtx, l1norm[tb_end] + 2 * rad);
  const uint64_t tile_size = 48 * 1024 / 4 / 2;
  __shared__ float shared[tile_size * 2];
  auto *const sh_x = shared;
  auto *const sh_y = shared + tile_size;
  const float ux = x[u], uy = y[u];
  uint64_t ans = 0;

  for (auto curr_ptr = range_start; curr_ptr < range_end;
       curr_ptr += tile_size) {
    uint64_t const curr_idx = curr_ptr - l1norm;
    uint64_t const i_stop = min(tile_size, range_end - curr_ptr);
    for (auto i = 0u; i < i_stop; ++i) {
      sh_x[i] = x[curr_idx + i];
      sh_y[i] = y[curr_idx + i];
    }
    __syncthreads();
    for (auto i = 0; i < i_stop; ++i) {
      ans += GDBSCAN::device_functions::square_dist(ux, uy, sh_x[i], sh_y[i]) <=
             rad * rad;
    }
  }
  num_nbs[vtx_mapper[u]] = ans - 1;
}
// Populate the actual neighbours array
__global__ void k_append_neighbours(float const *const x, float const *const y,
                                    float const *const l1norm,
                                    uint64_t const *const vtx_mapper,
                                    uint64_t const *const start_pos,
                                    const float rad, const uint64_t num_vtx,
                                    uint64_t *const neighbours) {
  uint64_t const u = threadIdx.x + blockIdx.x * blockDim.x;
  if (u >= num_vtx) return;
  // first vtx of current block.
  const uint64_t tb_start = blockIdx.x * blockDim.x;
  // last vtx of current block.
  const uint64_t tb_end = min(tb_start + blockDim.x, num_vtx - 1);
  // inclusive start
  const float *range_start = thrust::lower_bound(
      thrust::device, l1norm, l1norm + num_vtx, l1norm[tb_start] - 2 * rad);
  // exclusive end
  const float *range_end = thrust::upper_bound(
      thrust::device, l1norm, l1norm + num_vtx, l1norm[tb_end] + 2 * rad);
  const uint64_t tile_size = 48 * 1024 / 4 / 2;
  __shared__ float shared[tile_size * 2];
  auto *const sh_x = shared;
  auto *const sh_y = shared + tile_size;
  const float ux = x[u], uy = y[u];
  uint64_t upos = start_pos[vtx_mapper[u]];

  for (auto curr_ptr = range_start; curr_ptr < range_end;
       curr_ptr += tile_size) {
    uint64_t const curr_idx = curr_ptr - l1norm;
    uint64_t const i_stop = min(tile_size, range_end - curr_ptr);
    for (auto i = 0u; i < i_stop; ++i) {
      sh_x[i] = x[curr_idx + i];
      sh_y[i] = y[curr_idx + i];
    }
    __syncthreads();
    for (auto i = 0; i < i_stop; ++i) {
      uint64_t v = curr_idx + i;
      // TODO: fix me. one padding of each vertex range
      //      neighbours[upos] = vtx_mapper[v];
      //      upos += ((u != v) & GDBSCAN::device_functions::square_dist(
      //                              ux, uy, sh_x[v], sh_y[v]) <= rad * rad);
      if (u != v && GDBSCAN::device_functions::square_dist(
                        ux, uy, sh_x[i], sh_y[i]) <= rad * rad) {
        neighbours[upos++] = vtx_mapper[v];
      }
    }
  }
}
// Identify all the Core vtx.
__global__ void k_identify_cores(uint64_t const *const num_neighbours,
                                 DBSCAN::membership *const membership,
                                 const uint64_t num_vtx,
                                 const uint64_t min_pts) {
  uint64_t const u = threadIdx.x + blockIdx.x * blockDim.x;
  if (u >= num_vtx) return;
  membership[u] = DBSCAN::membership::Noise;
  if (num_neighbours[u] >= min_pts) membership[u] = DBSCAN::membership::Core;
}
// BFS kernel.
__global__ void k_bfs(bool *const visited, bool *const frontier,
                      uint64_t const *const num_nbs,
                      uint64_t const *const start_pos,
                      uint64_t const *const neighbours,
                      DBSCAN::membership const *const membership,
                      uint64_t num_vtx) {
  uint64_t const u = threadIdx.x + blockIdx.x * blockDim.x;
  if (u >= num_vtx) return;
  if (!frontier[u]) return;
  //  printf("\t\ttmark %lu visited and remove from frontier\n", u);
  frontier[u] = false;
  visited[u] = true;
  // Stop BFS if u is not Core.
  if (membership[u] != DBSCAN::membership::Core) return;
  uint64_t u_start = start_pos[u];
  //  printf("\t\t%lu has %lu nbs starting from idx %lu\n", u, num_nbs[u],
  //  u_start);
  for (uint64_t i = 0; i < num_nbs[u]; ++i) {
    uint64_t v = neighbours[u_start + i];
    frontier[v] = !visited[v];  // equal to if (!visited[v]) frontier[v] = true
  }
}
}  // namespace kernel_functions
}  // namespace GDBSCAN
#endif  // DBSCAN_GDBSCAN_KERNEL_FUNCTIONS_CUH
