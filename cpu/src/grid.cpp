//
// Created by will on 4/7/20.
//

#if defined(DBSCAN_TESTING)
#include <cassert>
#endif

#include <cmath>

#include "grid.h"
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

  // TODO: when GCC-10 is ready, use std::exclusive_scan with parallel exec.
  std::vector<std::thread> threads(num_threads_);
  for (uint8_t tid = 0; tid < num_threads_; ++tid) {
    threads[tid] = std::thread(
        [this, &xs, &ys](const uint8_t tid) {
          for (uint64_t vtx = tid; vtx < num_vtx_; vtx += num_threads_) {
            auto id = calc_cell_id_(xs[vtx], ys[vtx]);
            // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fsync-Builtins.html#g_t_005f_005fsync-Builtins
            __sync_fetch_and_add(grid_vtx_counter_.data() + id, 1);
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
  logger_->debug(
      DBSCAN::utils::print_vector("grid_start_pos_", grid_start_pos_));

  // make a local copy to record the write position.
  std::vector<uint64_t> temp(grid_start_pos_);

  for (uint8_t tid = 0; tid < num_threads_; ++tid) {
    threads[tid] = std::thread(
        [this, &xs, &ys, &temp](const uint8_t tid) {
          for (uint64_t vtx = tid; vtx < num_vtx_; vtx += num_threads_) {
            auto id = calc_cell_id_(xs[vtx], ys[vtx]);
            const auto pos = __sync_fetch_and_add(temp.data() + id, 1);
            // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fsync-Builtins.html#g_t_005f_005fsync-Builtins
            __sync_val_compare_and_swap(grid_.data() + pos, 0, vtx);
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
#if defined(DBSCAN_TESTING)
  assert(min_x_ < x);
  assert(x < max_x_);
  assert(min_y_ < y);
  assert(y < max_y_);
#endif
  uint64_t col_idx, row_idx;
  col_idx = std::floor((x - min_x_) / radius_) + 1;
  row_idx = std::floor((y - min_y_) / radius_) + 1;
#if defined(DBSCAN_TESTING)
  assert(0 < col_idx);
  assert(col_idx < grid_cols_);
  assert(0 < row_idx);
  assert(row_idx < grid_rows_);
#endif
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
  //  logger_->debug("nbs expected size {}", nbs.capacity());
  //  logger_->debug("{}",
  //                 DBSCAN::utils::print_vector("nbs @ begin of retrieve:",
  //                 nbs));
  for (auto i = 0u; i < grid_vtx_counter_[cell_id]; ++i) {
    const auto nb = grid_[grid_start_pos_[cell_id] + i];
    if (u != nb) nbs.push_back(nb);
  }
  //  logger_->debug("{}",
  //                 DBSCAN::utils::print_vector("nbs @ middle of retrieve:",
  //                 nbs));
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
  return nbs;
}
