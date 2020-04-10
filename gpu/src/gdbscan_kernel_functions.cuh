//
// Created by will on 2020-03-24.
//

#ifndef DBSCAN_GDBSCAN_KERNEL_FUNCTIONS_CUH
#define DBSCAN_GDBSCAN_KERNEL_FUNCTIONS_CUH

#include <membership.h>
#include <thrust/binary_search.h>

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
  uint64_t tb_start = blockIdx.x * blockDim.x;
  // last vtx of current block.
  uint64_t tb_end = std::min(tb_start + blockDim.x, num_vtx - 1);
  // inclusive start
  uint64_t range_start = thrust::lower_bound(
      thrust::device, l1norm, l1norm + num_vtx, l1norm[tb_start] - 2 * rad);
  // exclusive end
  uint64_t range_end = thrust::upper_bound(
      thrust::device, l1norm, l1norm + num_vtx, l1norm[tb_end] + 2 * rad);
  uint64_t range = range_end - range_start;
  uint64_t num_tiles = std::ceil(static_cast<float>(range) * 4 * 2 / 64 / 1024);
  uint64_t tile_size = std::ceil(static_cast<float>(range) / num_tiles);
  __shared__ float[tile_size] sh_x;
  __shared__ float[tile_size] sh_y;
  uint64_t ans = 0;
  for (auto curr = range_start; curr < range_end; curr += tile_size) {
    uint64_t i_stop = std::min(tile_size, range_end - curr);
    for (auto i = 0; i < i_stop; ++i) {
      sh_x[i] = x[range_start + curr + i];
      sh_y[i] = y[range_start + curr + i];
    }
    __sync_threads();
    for (auto i = 0; i < i_stop; ++i) {
      ans += GDBSCAN::device_functions::square_dist(
                 sh_x[u], sh_y[u], sh_x[range_start + curr + i],
                 sh_y[range_start + curr + i]) <= rad * rad
    }
  }
  num_nbs[vtx_mapper[u]] = ans;
}
// Populate the actual neighbours array
__global__ void k_append_neighbours(
    float const *const x, float const *const y,
    uint64_t const *const grid_vtx_counter,
    uint64_t const *const grid_start_pos, uint64_t const *const grid,
    const float minx, const float miny, const float radius,
    uint64_t const *const start_pos, uint64_t *const neighbours,
    const uint64_t num_vtx, const float rad_sq, const uint64_t grid_col_sz) {
  uint64_t const u = threadIdx.x + blockIdx.x * blockDim.x;
  if (u >= num_vtx) return;

  auto get_cell_id = GDBSCAN::device_functions::calc_cell_id;
  uint64_t cell_id = get_cell_id(x[u], y[u], minx, miny, radius, grid_col_sz);
  uint64_t left = cell_id - 1, btm_left = cell_id + grid_col_sz - 1,
           btm = cell_id + grid_col_sz, btm_right = cell_id + grid_col_sz + 1,
           right = cell_id + 1, top_right = cell_id - grid_col_sz + 1,
           top = cell_id - grid_col_sz, top_left = cell_id - grid_col_sz - 1;

  uint64_t upos = start_pos[u];
  auto dist = GDBSCAN::device_functions::square_dist;

  for (auto i = 0u; i < grid_vtx_counter[cell_id]; ++i) {
    const auto nb = grid[grid_start_pos[cell_id] + i];
    if (u != nb && dist(x[u], y[u], x[nb], y[nb]) <= rad_sq)
      neighbours[upos++] = nb;
  }
  for (auto i = 0u; i < grid_vtx_counter[left]; ++i) {
    const auto nb = grid[grid_start_pos[left] + i];
    if (dist(x[u], y[u], x[nb], y[nb]) <= rad_sq) neighbours[upos++] = nb;
  }
  for (auto i = 0u; i < grid_vtx_counter[btm_left]; ++i) {
    const auto nb = grid[grid_start_pos[btm_left] + i];
    if (dist(x[u], y[u], x[nb], y[nb]) <= rad_sq) neighbours[upos++] = nb;
  }
  for (auto i = 0u; i < grid_vtx_counter[btm]; ++i) {
    const auto nb = grid[grid_start_pos[btm] + i];
    if (dist(x[u], y[u], x[nb], y[nb]) <= rad_sq) neighbours[upos++] = nb;
  }
  for (auto i = 0u; i < grid_vtx_counter[btm_right]; ++i) {
    const auto nb = grid[grid_start_pos[btm_right] + i];
    if (dist(x[u], y[u], x[nb], y[nb]) <= rad_sq) neighbours[upos++] = nb;
  }
  for (auto i = 0u; i < grid_vtx_counter[right]; ++i) {
    const auto nb = grid[grid_start_pos[right] + i];
    if (dist(x[u], y[u], x[nb], y[nb]) <= rad_sq) neighbours[upos++] = nb;
  }
  for (auto i = 0u; i < grid_vtx_counter[top_right]; ++i) {
    const auto nb = grid[grid_start_pos[top_right] + i];
    if (dist(x[u], y[u], x[nb], y[nb]) <= rad_sq) neighbours[upos++] = nb;
  }
  for (auto i = 0u; i < grid_vtx_counter[top]; ++i) {
    const auto nb = grid[grid_start_pos[top] + i];
    if (dist(x[u], y[u], x[nb], y[nb]) <= rad_sq) neighbours[upos++] = nb;
  }
  for (auto i = 0u; i < grid_vtx_counter[top_left]; ++i) {
    const auto nb = grid[grid_start_pos[top_left] + i];
    if (dist(x[u], y[u], x[nb], y[nb]) <= rad_sq) neighbours[upos++] = nb;
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
