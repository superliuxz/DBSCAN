//
// Created by will on 4/7/20.
//

#ifndef DBSCAN_GRID_H
#define DBSCAN_GRID_H

#include <DBSCAN/utils.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "spdlog/spdlog.h"

namespace DBSCAN {
class Grid {
 public:
  Grid(float, float, float, float, float, uint64_t, uint8_t);
  void Construct(
      const std::vector<float, DBSCAN::utils::AlignedAllocator<float, 32>>&,
      const std::vector<float, DBSCAN::utils::AlignedAllocator<float, 32>>&);
  [[nodiscard]] std::vector<uint64_t> GetNeighbouringVtx(uint64_t, float,
                                                         float) const;

 private:
  float radius_;
  uint64_t num_vtx_;
  // grid
  float max_x_, max_y_, min_x_, min_y_;
  uint8_t num_threads_;
  uint64_t grid_rows_, grid_cols_;
  // Number of vertices in each grid cell. Size RxC.
  std::vector<uint64_t> grid_vtx_counter_;
  // Partial sum of |grid_vtx_counter_|. Size RxC.
  std::vector<uint64_t> grid_start_pos_;
  // grid_[i] = j means the ith vtx is in jth grid. Size num_vtx_.
  std::vector<uint64_t> grid_;
  std::shared_ptr<spdlog::logger> logger_ = nullptr;
  [[nodiscard]] uint64_t CalcCellId_(float x, float y) const;
};
}  // namespace DBSCAN

#endif  // DBSCAN_GRID_H
