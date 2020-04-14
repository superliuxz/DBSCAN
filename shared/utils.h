//
// Created by William Liu on 2020-01-31.
//

#ifndef DBSCAN_INCLUDE_UTILS_H_
#define DBSCAN_INCLUDE_UTILS_H_

#include <cmath>
#include <cstdlib>
#include <limits>
#include <new>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace DBSCAN {
namespace utils {
template <class T>
std::string print_vector(const std::string& vector_name,
                         std::vector<T> vector) {
  std::ostringstream oss;
  oss << vector_name << ": ";
  if (vector.empty()) {
    oss << "EMPTY!" << std::endl;
    return oss.str();
  }
  auto it = vector.cbegin();
  while (it != vector.cend() - 1) oss << *(it++) << ", ";
  oss << *it << std::endl;
  return oss.str();
}

// This allocator only allocates memory but never initializes anything. It's
// used to speed up Ea vector (which is huge).
// Copied from https://en.cppreference.com/w/cpp/named_req/Allocator
template <class T>
class NonConstructAllocator {
 public:
  typedef T value_type;
  NonConstructAllocator() = default;
  // clang-format off
  template <class U>
  constexpr explicit NonConstructAllocator (const NonConstructAllocator <U>&) noexcept {}
  template <class U>
  friend bool operator==(const NonConstructAllocator<T>&, const NonConstructAllocator<U>&) { return true; }
  template <class U>
  friend bool operator!=(const NonConstructAllocator<T>&, const NonConstructAllocator<U>&) { return false; }
  // clang-format on
  [[nodiscard]] T* allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_alloc();
    if (auto p = static_cast<T*>(std::malloc(n * sizeof(T)))) return p;
    throw std::bad_alloc();
  }
  void deallocate(T* p, std::size_t) noexcept { std::free(p); }
  // see https://stackoverflow.com/a/58985712 and its comments
  static void construct(T*, ...) {}
};

// Aligned allocator for SSE4.2/AVX
template <class T, size_t ALIGNMENT = 16>
class AlignedAllocator {
 public:
  typedef T value_type;
  AlignedAllocator() = default;
  // clang-format off
  template <class U>
  constexpr explicit AlignedAllocator (const AlignedAllocator <U>&) noexcept {}
  template <class U>
  friend bool operator==(const AlignedAllocator<T>&, const AlignedAllocator<U>&) { return true; }
  template <class U>
  friend bool operator!=(const AlignedAllocator<T>&, const AlignedAllocator<U>&) { return false; }
  // clang-format on
  [[nodiscard]] T* allocate(std::size_t n) {
    // If T's size < ALIGNMENT, each T occupies ALIGNMENT.
    // Else, each T occupies ALIGNMENT*ceil(T/ALIGNMENT).
    const size_t aligned_size =
        ALIGNMENT * ceil(sizeof(T) / static_cast<double>(ALIGNMENT));
    if (n > std::numeric_limits<std::size_t>::max() / aligned_size)
      throw std::bad_alloc();
    if (auto p = static_cast<T*>(aligned_alloc(ALIGNMENT, n * aligned_size)))
      return p;
    throw std::bad_alloc();
  }
  void deallocate(T* p, std::size_t) noexcept { std::free(p); }
  // see https://stackoverflow.com/q/12362363 why rebind is needed.
  template <class U>
  struct rebind {
    typedef AlignedAllocator<U, ALIGNMENT> other;
  };
};
}  // namespace utils
}  // namespace DBSCAN

#endif  // DBSCAN_INCLUDE_UTILS_H_
