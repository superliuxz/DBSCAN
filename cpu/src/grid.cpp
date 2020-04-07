//
// Created by will on 4/7/20.
//

#include "grid.h"

#include <cassert>
#include <cmath>

#include "spdlog/spdlog.h"

DBSCAN::Grid::Grid(float max_x, float max_y, float min_x, float min_y,
                   float radius, uint64_t num_vtx, uint8_t num_threads)
    : radius_(radius),
      num_vtx_(num_vtx),
      max_x_(max_x),
      max_y_(max_y),
      min_x_(min_x),
      min_y_(min_y),
      num_threads_(num_threads) {
  logger_ = spdlog::get("console");
  if (logger_ == nullptr) {
    throw std::runtime_error("logger not created!");
  }
  // "1+" prepends an empty col/row to the grid. The empty row/col includes
  // points {x in [-INF, min_x_), y in [-INF, min_y_)}.
  // "+1" appends an empty rol/col. The last row/col includes points
  // {x in [max_x_, INF), y in [max_y_, INF)}.
  grid_rows_ = 1 + std::ceil((max_y_ - min_y_) / radius) + 1;
  grid_cols_ = 1 + std::ceil((max_x_ - min_x_) / radius) + 1;
  grid_vtx_counter_.resize(grid_rows_ * grid_cols_);
  grid_.resize(num_vtx_);
}

void DBSCAN::Grid::construct_grid(
    const std::vector<float, DBSCAN::utils::AlignedAllocator<float, 32>>& xs,
    const std::vector<float, DBSCAN::utils::AlignedAllocator<float, 32>>& ys) {
  using namespace std::chrono;
  high_resolution_clock::time_point start = high_resolution_clock::now();

  std::vector<std::thread> threads(num_threads_);
  const uint64_t chunk = std::ceil(num_vtx_ / num_threads_);
  for (uint8_t tid = 0; tid < num_threads_; ++tid) {
    threads[tid] = std::thread(
        [this, &chunk, &xs, &ys](const uint8_t tid) {
          const uint64_t start = tid * chunk;
          const uint64_t end = std::min(start + chunk, num_vtx_);
          for (uint64_t vtx = start; vtx < end; ++vtx) {
            auto id = calc_cell_id_(xs[vtx], ys[vtx]);
            ++grid_vtx_counter_[id];
          }
        },
        tid);
  }
  for (auto& tr : threads) tr.join();

  logger_->debug(
      DBSCAN::utils::print_vector("grid_vtx_counter_", grid_vtx_counter_));
  grid_start_pos_.resize(grid_vtx_counter_.size(), 0);
  // TODO: exclusive_scan with GCC-10
  for (uint64_t i = 0; i < grid_vtx_counter_.size() - 1; ++i) {
    grid_start_pos_[i + 1] = grid_start_pos_[i] + grid_vtx_counter_[i];
  }
  // make a local copy to record the write position.
  std::vector<uint64_t> temp(grid_start_pos_);
  logger_->debug(
      DBSCAN::utils::print_vector("grid_start_pos_", grid_start_pos_));

  for (uint8_t tid = 0; tid < num_threads_; ++tid) {
    threads[tid] = std::thread(
        [this, &chunk, &xs, &ys, &temp](const uint8_t tid) {
          const uint64_t start = tid * chunk;
          const uint64_t end = std::min(start + chunk, num_vtx_);
          for (uint64_t vtx = start; vtx < end; ++vtx) {
            auto id = calc_cell_id_(xs[vtx], ys[vtx]);
            grid_[temp[id]++] = vtx;
          }
        },
        tid);
  }
  for (auto& tr : threads) tr.join();
  logger_->debug(DBSCAN::utils::print_vector("grid", grid_));
  duration<double> time_spent =
      duration_cast<duration<double>>(high_resolution_clock::now() - start);
  logger_->info("construct_grid takes {} seconds", time_spent.count());
}

uint64_t DBSCAN::Grid::calc_cell_id_(float x, float y) const {
  // because of the offset, x/y should never be equal to min/max.
  assert(min_x_ < x);
  assert(x < max_x_);
  assert(min_y_ < y);
  assert(y < max_y_);
  uint64_t col_idx, row_idx;
  col_idx = std::floor((x - min_x_) / radius_) + 1;
  row_idx = std::floor((y - min_y_) / radius_) + 1;
  assert(0 < col_idx);
  assert(col_idx < grid_cols_);
  assert(0 < row_idx);
  assert(row_idx < grid_rows_);
  return row_idx * grid_cols_ + col_idx;
}

std::vector<uint64_t> DBSCAN::Grid::retrieve_vtx_from_nb_cells(uint64_t u,
                                                               float ux,
                                                               float uy) const {
  uint64_t cell_id = calc_cell_id_(ux, uy);
  uint64_t left = cell_id - 1, btm_left = cell_id + grid_cols_ - 1,
           btm = cell_id + grid_cols_, btm_right = cell_id + grid_cols_ + 1,
           right = cell_id + 1, top_right = cell_id - grid_cols_ + 1,
           top = cell_id - grid_cols_, top_left = cell_id - grid_cols_ - 1;
  std::vector<uint64_t> nbs;
  nbs.reserve(grid_vtx_counter_[cell_id] + grid_vtx_counter_[left] +
              grid_vtx_counter_[btm_left] + grid_vtx_counter_[btm] +
              grid_vtx_counter_[btm_right] + grid_vtx_counter_[right] +
              grid_vtx_counter_[top_right] + grid_vtx_counter_[top] +
              grid_vtx_counter_[top_left]);
  //  printf("nbs expected size %lu\n", nbs.capacity());
  //  printf("%s", DBSCAN::utils::print_vector("nbs @ begin of retrieve:",
  //  nbs).c_str());
  for (auto i = 0u; i < grid_vtx_counter_[cell_id]; ++i) {
    const auto nb = grid_[grid_start_pos_[cell_id] + i];
    //    printf("vtx %lu nb %lu cellid %lu startpos %lu\n", u, nb, cell_id,
    //           grid_start_pos_[cell_id]);
    if (u != nb) nbs.push_back(nb);
  }
  //  printf("%s", DBSCAN::utils::print_vector("nbs @ middle of retrieve:",
  //  nbs).c_str());
  for (auto i = 0u; i < grid_vtx_counter_[left]; ++i)
    nbs.push_back(grid_[grid_start_pos_[left] + i]);
  for (auto i = 0u; i < grid_vtx_counter_[btm_left]; ++i)
    nbs.push_back(grid_[grid_start_pos_[btm_left] + i]);
  for (auto i = 0u; i < grid_vtx_counter_[btm]; ++i)
    nbs.push_back(grid_[grid_start_pos_[btm] + i]);
  for (auto i = 0u; i < grid_vtx_counter_[btm_right]; ++i)
    nbs.push_back(grid_[grid_start_pos_[btm_right] + i]);
  for (auto i = 0u; i < grid_vtx_counter_[right]; ++i)
    nbs.push_back(grid_[grid_start_pos_[right] + i]);
  for (auto i = 0u; i < grid_vtx_counter_[top_right]; ++i)
    nbs.push_back(grid_[grid_start_pos_[top_right] + i]);
  for (auto i = 0u; i < grid_vtx_counter_[top]; ++i)
    nbs.push_back(grid_[grid_start_pos_[top] + i]);
  for (auto i = 0u; i < grid_vtx_counter_[top_left]; ++i)
    nbs.push_back(grid_[grid_start_pos_[top_left] + i]);
  //  printf("%s", DBSCAN::utils::print_vector("nbs @ end of retrieve:",
  //  nbs).c_str());
  return nbs;
}
