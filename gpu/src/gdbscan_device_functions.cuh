//
// Created by will on 2020-03-24.
//

#ifndef DBSCAN_DEVICE_FUNCTIONS_CUH
#define DBSCAN_DEVICE_FUNCTIONS_CUH

namespace GDBSCAN {
namespace device_functions {
__device__ float square_dist(const float x1, const float y1, const float x2,
                             const float y2) {
  return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}
__device__ uint64_t calc_cell_id(const float x, const float y,
                                 const float min_x, const float min_y,
                                 const float radius,
                                 const uint64_t grid_col_size) {
  uint64_t col_idx = std::floor((x - min_x) / radius) + 1;
  uint64_t row_idx = std::floor((y - min_y) / radius) + 1;
  return row_idx * grid_col_size + col_idx;
}
}  // namespace device_functions
}  // namespace GDBSCAN

#endif  // DBSCAN_DEVICE_FUNCTIONS_CUH
