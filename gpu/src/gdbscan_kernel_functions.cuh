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
                          uint64_t const *const vtx_mapper, const float rad,
                          const uint64_t num_vtx, uint64_t *const num_nbs) {
  uint64_t const u = threadIdx.x + blockIdx.x * blockDim.x;
  if (u >= num_vtx) return;
  // first vtx of current block.
  const uint64_t tb_start = blockIdx.x * blockDim.x;
  // last vtx of current block.
  const uint64_t tb_end = min(tb_start + blockDim.x, num_vtx) - 1;
  // inclusive start
  const float *possible_range_start = thrust::lower_bound(
      thrust::device, l1norm, l1norm + num_vtx, l1norm[tb_start] - 2 * rad);
  // exclusive end
  const float *possible_range_end = thrust::upper_bound(
      thrust::device, l1norm, l1norm + num_vtx, l1norm[tb_end] + 2 * rad);
  const uint64_t tile_size = 48 * 1024 / 4 / (1 + 1);
  uint64_t const num_threads = tb_end - tb_start;
  // first half of shared stores Xs; second half stores Ys.
  __shared__ float shared[tile_size * (1 + 1)];
  auto *const sh_x = shared;
  auto *const sh_y = shared + tile_size;
  const float ux = x[u], uy = y[u];
  uint64_t ans = 0;

  for (auto curr_ptr = possible_range_start; curr_ptr < possible_range_end;
       curr_ptr += tile_size) {
    // curr_ptr's index
    uint64_t const curr_idx = curr_ptr - l1norm;
    // current range; might be less than tile_size.
    uint64_t const curr_range = min(tile_size, possible_range_end - curr_ptr);
    // each thread updates sub_range number of Xs and Ys.
    uint64_t const sub_range =
        std::ceil(curr_range / static_cast<float>(num_threads));
    uint64_t const i_start = sub_range * threadIdx.x;
    uint64_t const i_stop = min(i_start + sub_range, curr_range);
    for (auto i = i_start; i < i_stop; ++i) {
      sh_x[i] = x[curr_idx + i];
      sh_y[i] = y[curr_idx + i];
    }
    __syncthreads();
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
                                    uint64_t const *const vtx_mapper,
                                    uint64_t const *const start_pos,
                                    const float rad, const uint64_t num_vtx,
                                    uint64_t *const neighbours) {
  uint64_t const u = threadIdx.x + blockIdx.x * blockDim.x;
  if (u >= num_vtx) return;
  // first vtx of current block.
  const uint64_t tb_start = blockIdx.x * blockDim.x;
  // last vtx of current block.
  const uint64_t tb_end = min(tb_start + blockDim.x, num_vtx) - 1;
  // inclusive start
  const float *possible_range_start = thrust::lower_bound(
      thrust::device, l1norm, l1norm + num_vtx, l1norm[tb_start] - 2 * rad);
  // exclusive end
  const float *possible_range_end = thrust::upper_bound(
      thrust::device, l1norm, l1norm + num_vtx, l1norm[tb_end] + 2 * rad);
  // different from previous kernel, here the shared array is tri-partitioned,
  // because of the frequent access to vtx_mapper. The vtx_mapper's partition
  // size is twice large since uint64_t is 8 bytes whereas float is 4.
  const uint64_t tile_size = 48 * 1024 / 4 / (1 + 1 + 2);
  uint64_t const num_threads = tb_end - tb_start;
  __shared__ float shared[tile_size * (1 + 1 + 2)];
  auto *const sh_x = shared;
  auto *const sh_y = shared + tile_size;
  auto *const sh_vtx_mapper = (uint64_t *)(sh_y + tile_size);
  const float ux = x[u], uy = y[u];
  uint64_t upos = start_pos[vtx_mapper[u]];

  for (auto curr_ptr = possible_range_start; curr_ptr < possible_range_end;
       curr_ptr += tile_size) {
    // curr_ptr's index
    uint64_t const curr_idx = curr_ptr - l1norm;
    // current range; might be less than tile_size.
    uint64_t const curr_range = min(tile_size, possible_range_end - curr_ptr);
    // each thread updates sub_range number of Xs and Ys.
    uint64_t const sub_range =
        std::ceil(curr_range / static_cast<float>(num_threads));
    uint64_t const i_start = sub_range * threadIdx.x;
    uint64_t const i_stop = min(i_start + sub_range, curr_range);
    for (auto i = i_start; i < i_stop; ++i) {
      sh_x[i] = x[curr_idx + i];
      sh_y[i] = y[curr_idx + i];
      sh_vtx_mapper[i] = vtx_mapper[curr_idx + i];
    }
    __syncthreads();
    for (auto j = 0; j < curr_range; ++j) {
      // TODO: fix me. one padding of each vertex range
      //      neighbours[upos] = vtx_mapper[v];
      //      upos += ((u != v) & GDBSCAN::device_functions::square_dist(
      //                              ux, uy, sh_x[v], sh_y[v]) <= rad * rad);
      if (u != curr_idx + j && GDBSCAN::device_functions::square_dist(
                                   ux, uy, sh_x[j], sh_y[j]) <= rad * rad) {
        neighbours[upos++] = sh_vtx_mapper[j];
      }
    }
    __syncthreads();
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
__global__ void k_identify_cores(uint64_t const *const num_neighbours,
                                 DBSCAN::membership *const membership,
                                 const uint64_t num_vtx,
                                 const uint64_t min_pts) {
  uint64_t const u = threadIdx.x + blockIdx.x * blockDim.x;
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
