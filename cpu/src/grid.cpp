//
// Created by will on 4/7/20.
//

#if defined(DBSCAN_TESTING)
#include <cassert>
#endif

#include <cmath>
#include <execution>
#include <functional>

#include "grid.h"
#include "spdlog/spdlog.h"

DBSCAN::Grid::Grid(const float max_x, const float max_y, const float min_x,
                   const float min_y, const float radius,
                   const uint64_t num_vtx, const uint8_t num_threads)
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

void DBSCAN::Grid::Construct(
    const std::vector<float, DBSCAN::utils::AlignedAllocator<float, 32>>& xs,
    const std::vector<float, DBSCAN::utils::AlignedAllocator<float, 32>>& ys) {
  using namespace std::chrono;
  high_resolution_clock::time_point start = high_resolution_clock::now();

  std::vector<std::thread> threads(num_threads_);
  for (uint8_t tid = 0; tid < num_threads_; ++tid) {
    threads[tid] = std::thread(
        [this, &xs, &ys](const uint8_t tid) {
          for (uint64_t vtx = tid; vtx < num_vtx_; vtx += num_threads_) {
            const auto id = CalcCellId_(xs[vtx], ys[vtx]);
            // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fsync-Builtins.html#g_t_005f_005fsync-Builtins
            // Fetch grid_vtx_counter_[id] and add one, synchronized.
            __sync_fetch_and_add(grid_vtx_counter_.data() + id, 1);
          }
        },
        tid);
  }
  for (auto& tr : threads) tr.join();

  logger_->debug(
      DBSCAN::utils::print_vector("grid_vtx_counter_", grid_vtx_counter_));

  grid_start_pos_.resize(grid_vtx_counter_.size(), 0);
  // exclusive_scan on |grid_vtx_counter_| and store the results in
  // |grid_start_pos_|.
  std::exclusive_scan(std::execution::par_unseq, grid_vtx_counter_.cbegin(),
                      grid_vtx_counter_.cend(), grid_start_pos_.begin(), 0,
                      std::plus<uint64_t>{});
  logger_->debug(
      DBSCAN::utils::print_vector("grid_start_pos_", grid_start_pos_));

  // make a local copy to record the write position.
  std::vector<uint64_t> temp(grid_start_pos_);

  for (uint8_t tid = 0; tid < num_threads_; ++tid) {
    threads[tid] = std::thread(
        [this, &xs, &ys, &temp](const uint8_t tid) {
          for (uint64_t vtx = tid; vtx < num_vtx_; vtx += num_threads_) {
            const auto id = CalcCellId_(xs[vtx], ys[vtx]);
            // Synchronized: pos = temp[id]++;
            const auto pos = __sync_fetch_and_add(temp.data() + id, 1);
            // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fsync-Builtins.html#g_t_005f_005fsync-Builtins
            // If grid_[pos] is 0, set grid_[pos] to vtx. Synchronized.
            __sync_val_compare_and_swap(grid_.data() + pos, 0, vtx);
          }
        },
        tid);
  }
  for (auto& tr : threads) tr.join();
  logger_->debug(DBSCAN::utils::print_vector("grid", grid_));
  duration<double> time_spent =
      duration_cast<duration<double>>(high_resolution_clock::now() - start);
  logger_->info("Construct takes {} seconds", time_spent.count());
}

uint64_t DBSCAN::Grid::CalcCellId_(float x, float y) const {
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

std::vector<uint64_t> DBSCAN::Grid::GetNeighbouringVtx(const uint64_t u,
                                                       const float ux,
                                                       const float uy) const {
  const uint64_t cell_id = CalcCellId_(ux, uy);
  // clang-format off
  const uint64_t btm_left = cell_id + grid_cols_ - 1,
                 left = cell_id - 1,
                 right = cell_id + 1,
                 top_left = cell_id - grid_cols_ - 1;
  std::vector<uint64_t> nbs;
  nbs.reserve(grid_vtx_counter_[top_left] + grid_vtx_counter_[top_left + 1] + grid_vtx_counter_[top_left + 2] + /* top row */
              grid_vtx_counter_[left] + grid_vtx_counter_[cell_id] + grid_vtx_counter_[right] + /* current row */
              grid_vtx_counter_[btm_left] + grid_vtx_counter_[btm_left + 1] + grid_vtx_counter_[btm_left + 2] /* btm row */);
  // top row
  for (auto col = 0u; col < 3; ++col) {
    nbs.insert(nbs.end(),
               grid_.cbegin() + grid_start_pos_[top_left + col],
               grid_.cbegin() + grid_start_pos_[top_left + col] + grid_vtx_counter_[top_left + col]);
  }
  // current row
  nbs.insert(nbs.end(),
             grid_.cbegin() + grid_start_pos_[left],
             grid_.cbegin() + grid_start_pos_[left] + grid_vtx_counter_[left]);
  for (auto i = 0u; i < grid_vtx_counter_[cell_id]; ++i) {
    const auto nb = grid_[grid_start_pos_[cell_id] + i];
    if (u != nb) nbs.push_back(nb);
  }
  nbs.insert(nbs.end(),
             grid_.cbegin() + grid_start_pos_[right],
             grid_.cbegin() + grid_start_pos_[right] + grid_vtx_counter_[right]);
  // btm row
  for (auto col = 0u; col < 3; ++col) {
    nbs.insert(nbs.end(),
               grid_.cbegin() + grid_start_pos_[btm_left + col],
               grid_.cbegin() + grid_start_pos_[btm_left + col] + grid_vtx_counter_[btm_left + col]);
  }
  // clang-format on
  return nbs;
}
