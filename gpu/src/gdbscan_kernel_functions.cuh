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
// Write cell_id_array and vtx_idx_array. One thread per vtx.
__global__ void k_populate_cell_id_array(float const *const x,
                                         float const *const y, const float minx,
                                         const float miny, const float radius,
                                         const uint64_t grid_col_sz,
                                         uint64_t *const cell_id_array,
                                         uint64_t *const vtx_idx_array,
                                         const uint64_t num_vtx) {
  uint64_t const u = threadIdx.x + blockIdx.x * blockDim.x;
  if (u >= num_vtx) return;
  auto get_cell_id = GDBSCAN::device_functions::calc_cell_id;
  const auto id = get_cell_id(x[u], y[u], minx, miny, radius, grid_col_sz);
  cell_id_array[u] = id;
  vtx_idx_array[u] = u;
}
// Calculate the number of vertices of each cell using binary search. One
// thread per cell.
__global__ void k_calc_grid_vtx_counter(uint64_t const *const cell_id_array,
                                        uint64_t *const grid_vtx_counter,
                                        const uint64_t num_vtx,
                                        const uint64_t grid_sz) {
  uint64_t const cell_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (cell_id >= grid_sz) return;
  const auto p = thrust::equal_range(thrust::device, cell_id_array,
                                     cell_id_array + num_vtx, cell_id);
  grid_vtx_counter[cell_id] = p.second - p.first;
}
// Populate grid. One thread per cell.
__global__ void k_populate_grid(uint64_t const *const cell_id_array,
                                uint64_t const *const vtx_idx_array,
                                uint64_t const *const grid_start_pos,
                                uint64_t *const grid, const uint64_t num_vtx,
                                const uint64_t grid_sz) {
  uint64_t const cell_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (cell_id >= grid_sz) return;
  const auto p = thrust::equal_range(thrust::device, cell_id_array,
                                     cell_id_array + num_vtx, cell_id);
  const auto start = p.first - cell_id_array;
  const auto end = start + p.second - p.first;
  //  printf("cell_id %lu/%lu start %lu end %lu start_pos %lu\n", cell_id,
  //  grid_sz,
  //         start, end, grid_start_pos[cell_id]);
  auto out_ptr = grid + grid_start_pos[cell_id];
  for (auto ptr = vtx_idx_array + start; ptr != vtx_idx_array + end;
       ++ptr, ++out_ptr) {
    //    printf("\twrite %lu to %lu\n", ptr, out_ptr);
    *out_ptr = *ptr;
  }
}
// Calculate the number of neighbours of each vertex
__global__ void k_num_nbs(float const *const x, float const *const y,
                          uint64_t *const num_nbs, const float rad_sq,
                          const uint64_t num_vtx) {
  uint64_t const u = threadIdx.x + blockIdx.x * blockDim.x;
  if (u >= num_vtx) return;
  num_nbs[u] = 0;
  auto dist = GDBSCAN::device_functions::square_dist;
  for (auto v = 0u; v < num_vtx; ++v) {
    if (u != v && dist(x[u], y[u], x[v], y[v]) <= rad_sq) ++num_nbs[u];
  }
}
// Populate the actual neighbours array
__global__ void k_append_neighbours(float const *const x, float const *const y,
                                    uint64_t const *const start_pos,
                                    uint64_t *const neighbours,
                                    const uint64_t num_vtx,
                                    const float rad_sq) {
  uint64_t const u = threadIdx.x + blockIdx.x * blockDim.x;
  if (u >= num_vtx) return;
  uint64_t upos = start_pos[u];
  auto dist = GDBSCAN::device_functions::square_dist;
  for (uint64_t v = 0u; v < num_vtx; ++v) {
    if (u != v && dist(x[u], y[u], x[v], y[v]) <= rad_sq)
      neighbours[upos++] = v;
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
