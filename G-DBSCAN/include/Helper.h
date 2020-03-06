//
// Created by William Liu on 2020-01-31.
//

#ifndef GDBSCAN_INCLUDE_HELPER_H_
#define GDBSCAN_INCLUDE_HELPER_H_

#include <cstdlib>
#include <limits>
#include <new>
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

// This allocator only allocates memory but never initializes anything. It's
// used to speed up Ea vector (which is huge).
// Copied from https://en.cppreference.com/w/cpp/named_req/Allocator
template <class T>
struct NonConstructAllocator {
  typedef T value_type;
  T* allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_alloc();
    if (auto p = static_cast<T*>(std::malloc(n * sizeof(T)))) return p;
    throw std::bad_alloc();
  }
  void deallocate(T* p, std::size_t) noexcept { std::free(p); }
  // see https://stackoverflow.com/a/58985712 and its comments
  static void construct(T*, ...) {}
};
}  // namespace GDBSCAN::helper

#endif  // GDBSCAN_INCLUDE_HELPER_H_
