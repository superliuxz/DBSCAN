//
// Created by William Liu on 2020-01-21.
//
#ifndef PAGERANK_INCLUDE_MATRIX_H_
#define PAGERANK_INCLUDE_MATRIX_H_

#include <sstream>
#include <vector>

namespace pagerank {
class Matrix {
 public:
  // ctor
  Matrix(size_t n_row, size_t n_col)
      : n_row_(n_row), n_col_(n_col), data_(n_row * n_col, 0.0f) {
  }
  // setter
  float &operator()(const size_t row, const size_t col) {
    if (row >= this->n_row_ || col >= this->n_col_) {
      std::ostringstream err;
      this->format_oob_err(row, col, err);
      throw std::runtime_error(err.str());
    }
    return data_[this->n_col_ * row + col];
  }
  // getter
  float operator()(const size_t row, const size_t col) const {
    if (row >= this->n_row_ || col >= this->n_col_) {
      std::ostringstream err;
      this->format_oob_err(row, col, err);
      throw std::runtime_error(err.str());
    }
    return data_[this->n_col_ * row + col];
  }
  // matrix multiplication, this * other.
  Matrix operator*(const Matrix &other) const {
    assert(this->n_col_ == other.n_row_);

    size_t rows = this->n_row_, cols = other.n_col_;
    Matrix new_data(rows, cols);

    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < cols; ++col) {
        for (size_t curr = 0; curr < this->n_col_; ++curr) {
          new_data(row, col) += this->operator()(row, curr) * other(curr, col);
        }
      }
    }
    return new_data;
  }
  // matrix * scalar
  Matrix operator*(const float &factor) const {
    Matrix new_data(this->n_row_, this->n_col_);
    for (size_t r = 0; r < this->n_row_; ++r) {
      for (size_t c = 0; c < this->n_col_; ++c) {
        new_data(r, c) = this->operator()(r, c) * factor;
      }
    }
    return new_data;
  }
  // matrix / scalar
  Matrix operator/(const float &factor) const {
    Matrix new_data(this->n_row_, this->n_col_);
    for (size_t r = 0; r < this->n_row_; ++r) {
      for (size_t c = 0; c < this->n_col_; ++c) {
        new_data(r, c) = this->operator()(r, c) / factor;
      }
    }
    return new_data;
  }
  // matrix + scalar
  Matrix operator+(const float &factor) const {
    Matrix new_data(this->n_row_, this->n_col_);
    for (size_t r = 0; r < this->n_row_; ++r) {
      for (size_t c = 0; c < this->n_col_; ++c) {
        new_data(r, c) = this->operator()(r, c) + factor;
      }
    }
    return new_data;
  }
  // pipe
  friend std::ostream &operator<<(std::ostream &stream, const Matrix &m) {
    stream << std::fixed << std::setprecision(4);
    for (size_t r = 0; r < m.n_row_; ++r) {
      stream << m(r, 0);
      for (size_t c = 1; c < m.n_col_; ++c) {
        stream << " " << m(r, c);
      }
      stream << std::endl;
    }
    return stream;
  }

 private:
  void format_oob_err(const size_t &i,
                      const size_t &j,
                      std::ostringstream &err) const {
    err << "idx (" << i << "," << j << ") out of bound! (" << n_row_
        << "," << n_col_ << ")";
  }
  std::vector<float> data_;
  size_t n_row_;
  size_t n_col_;
};
} // namespace pagerank

#endif //PAGERANK_INCLUDE_MATRIX_H_
