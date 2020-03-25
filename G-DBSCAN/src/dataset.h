//
// Created by William Liu on 2020-03-06.
//

#ifndef DBSCAN_INCLUDE_DATASET_H_
#define DBSCAN_INCLUDE_DATASET_H_

#include <cmath>
#include <vector>

#include "utils.h"

namespace DBSCAN::input_type {
struct TwoDimPoints {
  std::vector<float, DBSCAN::utils::AlignedAllocator<float, 32>> d1, d2;
  explicit TwoDimPoints(size_t num_nodes) : d1(num_nodes), d2(num_nodes) {}
  static inline float euclidean_distance_square(const float& px,
                                                const float& py,
                                                const float& qx,
                                                const float& qy) {
    return (px - qx) * (px - qx) + (py - qy) * (py - qy);
  }
};
}  // namespace DBSCAN::input_type

#endif  // DBSCAN_INCLUDE_DATASET_H_
