//
// Created by will on 2020-03-24.
//

#ifndef DBSCAN_DEVICE_FUNCTIONS_CUH
#define DBSCAN_DEVICE_FUNCTIONS_CUH

namespace GDBSCAN {
namespace device_functions {
__device__ float square_dist(const float &x1, const float &y1, const float &x2,
                             const float &y2) {
  return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}
}  // namespace device_functions
}  // namespace GDBSCAN

#endif  // DBSCAN_DEVICE_FUNCTIONS_CUH
