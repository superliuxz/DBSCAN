//
// Created by William Liu on 2020-01-31.
//

#ifndef GDBSCAN_INCLUDE_HELPER_H_
#define GDBSCAN_INCLUDE_HELPER_H_

#include <sstream>
#include <string>
#include <vector>

namespace GDBSCAN::helper {
template<class T>
static std::string print_vector(const std::string &vector_name,
                                std::vector<T> vector) {
  std::ostringstream oss;
  oss << vector_name << ": ";
  const auto &it = vector.cbegin();
  while (it != vector.cend() - 1) {
    std::cout << *it << ", ";
    std::advance(it);
  }
  std::cout << *(it++) << std::endl;
  return oss.str();
}
}

#endif //GDBSCAN_INCLUDE_HELPER_H_
