//
// Created by will on 2020-03-24.
//

#ifndef DBSCAN_GDBSCAN_KERNEL_FUNCTIONS_CUH
#define DBSCAN_GDBSCAN_KERNEL_FUNCTIONS_CUH

namespace GDBSCAN {
namespace kernel_functions {
// Calculate the number of neighbours of each node
__global__ void k_num_nbs(float const *const x, float const *const y,
                          uint64_t *const num_nbs, const float rad_sq,
                          const uint64_t num_nodes) {
  uint64_t const u = threadIdx.x + blockIdx.x * blockDim.x;
  if (u >= num_nodes) return;
  num_nbs[u] = 0;
  for (auto v = 0u; v < num_nodes; ++v) {
    if (u != v && GDBSCAN::device_functions::square_dist(x[u], y[u], x[v],
                                                         y[v]) <= rad_sq)
      ++num_nbs[u];
  }
}
// Populate the actual neighbours array
__global__ void k_append_neighbours(float const *const x, float const *const y,
                                    uint64_t const *const start_pos,
                                    uint64_t *const neighbours,
                                    const uint64_t num_nodes,
                                    const float rad_sq) {
  uint64_t const u = threadIdx.x + blockIdx.x * blockDim.x;
  if (u >= num_nodes) return;
  uint64_t upos = start_pos[u];
  for (uint64_t v = 0u; v < num_nodes; ++v) {
    if (u != v && GDBSCAN::device_functions::square_dist(x[u], y[u], x[v],
                                                         y[v]) <= rad_sq)
      neighbours[upos++] = v;
  }
}
}  // namespace kernel_functions
}  // namespace GDBSCAN
#endif  // DBSCAN_GDBSCAN_KERNEL_FUNCTIONS_CUH
