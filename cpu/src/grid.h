//
// Created by will on 4/7/20.
//

#ifndef DBSCAN_GRID_H
#define DBSCAN_GRID_H

#include <utils.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "spdlog/spdlog.h"

namespace DBSCAN {
class Grid {
 public:
  Grid(float, float, float, float, float, uint64_t);
  void construct_grid(
      const std::vector<float, DBSCAN::utils::AlignedAllocator<float, 32>>&,
      const std::vector<float, DBSCAN::utils::AlignedAllocator<float, 32>>&);
  [[nodiscard]] std::vector<uint64_t> retrieve_vtx_from_nb_cells(uint64_t,
                                                                 float,
                                                                 float) const;

 private:
  float radius_;
  uint64_t num_vtx_;
  // grid
  float max_x_, max_y_, min_x_, min_y_;
  uint64_t grid_rows_, grid_cols_;
  std::vector<uint64_t> grid_vtx_counter_, grid_start_pos_, grid_;
  std::shared_ptr<spdlog::logger> logger_ = nullptr;
  [[nodiscard]] uint64_t calc_cell_id_(float, float) const;
};
}  // namespace DBSCAN

#endif  // DBSCAN_GRID_H
