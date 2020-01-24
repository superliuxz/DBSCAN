//
// Created by William Liu on 2020-01-24.
//

#ifndef GDBSCAN_INCLUDE_DIMENSION_H_
#define GDBSCAN_INCLUDE_DIMENSION_H_

#include <cmath>

namespace GDBSCAN::dimension {

class TwoD {
 public:
  TwoD() {}
  TwoD(float x, float y) : x_(x), y_(y) {}
  double operator-(const TwoD &o) const {
    return std::sqrt(std::pow(x_ - o.x_, 2) + std::pow(y_ - o.y_, 2));
  }
  bool operator==(const TwoD &o) const {
    return x_ == o.x_ && y_ == o.y_;
  }
 private:
  float x_, y_;
};
} // namespace GDBSCAN::dimension

#endif //GDBSCAN_INCLUDE_DIMENSION_H_
