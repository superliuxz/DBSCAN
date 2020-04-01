//
// Created by will on 2020-03-24.
//

#ifndef DBSCAN_GDBSCAN_KERNEL_FUNCTIONS_CUH
#define DBSCAN_GDBSCAN_KERNEL_FUNCTIONS_CUH

#include <membership.h>

#include "gdbscan_device_functions.cuh"

namespace GDBSCAN {
namespace kernel_functions {
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
