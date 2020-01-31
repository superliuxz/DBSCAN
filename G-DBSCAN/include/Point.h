//
// Created by William Liu on 2020-01-24.
//

#ifndef GDBSCAN_INCLUDE_POINT_H_
#define GDBSCAN_INCLUDE_POINT_H_

#include <cmath>

namespace GDBSCAN::point {

class EuclideanTwoD {
 public:
  EuclideanTwoD() {}
  EuclideanTwoD(float x, float y) : x_(x), y_(y) {}
  double operator-(const EuclideanTwoD &o) const {
    return std::sqrt(std::pow(x_ - o.x_, 2) + std::pow(y_ - o.y_, 2));
  }
  // two floats; each float is 4 bytes.
  static size_t size() { return 8; }
#ifdef TESTING
  bool operator==(const EuclideanTwoD &o) const {
    return x_ == o.x_ && y_ == o.y_;
  }
#endif
 private:
  float x_, y_;
};
} // namespace GDBSCAN::point

#endif //GDBSCAN_INCLUDE_POINT_H_
