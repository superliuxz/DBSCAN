//
// Created by William Liu on 2020-01-31.
//

#ifndef GDBSCAN_INCLUDE_HELPER_H_
#define GDBSCAN_INCLUDE_HELPER_H_

#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace GDBSCAN::helper {
template <class T>
std::string print_vector(const std::string& vector_name,
                         std::vector<T> vector) {
  std::ostringstream oss;
  oss << vector_name << ": ";
  auto it = vector.cbegin();
  while (it != vector.cend() - 1) {
    oss << *it << ", ";
    std::advance(it);
  }
  oss << *it << std::endl;
  return oss.str();
}
}  // namespace GDBSCAN::helper

#endif  // GDBSCAN_INCLUDE_HELPER_H_
