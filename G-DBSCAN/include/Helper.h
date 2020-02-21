//
// Created by William Liu on 2020-01-31.
//

#ifndef GDBSCAN_INCLUDE_HELPER_H_
#define GDBSCAN_INCLUDE_HELPER_H_

#include <sstream>
#include <string>
#include <vector>

namespace GDBSCAN::helper {
template <class T>
std::string print_vector(const std::string &vector_name,
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
// clang-format off
const uint64_t m1  = 0x5555555555555555; //binary: 0101...
const uint64_t m2  = 0x3333333333333333; //binary: 00110011..
const uint64_t m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
const uint64_t h01 = 0x0101010101010101; //the sum of 256 to the power of 0,1,2,3...
inline size_t popcount64(uint64_t x) {
  x -= (x >> 1) & m1;              // put count of each 2 bits into those 2 bits
  x = (x & m2) + ((x >> 2) & m2);  // put count of each 4 bits into those 4 bits
  x = (x + (x >> 4)) & m4;         // put count of each 8 bits into those 8 bits
  return (x * h01) >> 56;  // returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ...
}
// clang-format on
const uint64_t lookup[64] = {
    1ull << 0,  1ull << 1,  1ull << 2,  1ull << 3,  1ull << 4,  1ull << 5,
    1ull << 6,  1ull << 7,  1ull << 8,  1ull << 9,  1ull << 10, 1ull << 11,
    1ull << 12, 1ull << 13, 1ull << 14, 1ull << 15, 1ull << 16, 1ull << 17,
    1ull << 18, 1ull << 19, 1ull << 20, 1ull << 21, 1ull << 22, 1ull << 23,
    1ull << 24, 1ull << 25, 1ull << 26, 1ull << 27, 1ull << 28, 1ull << 29,
    1ull << 30, 1ull << 31, 1ull << 32, 1ull << 33, 1ull << 34, 1ull << 35,
    1ull << 36, 1ull << 37, 1ull << 38, 1ull << 39, 1ull << 40, 1ull << 41,
    1ull << 42, 1ull << 43, 1ull << 44, 1ull << 45, 1ull << 46, 1ull << 47,
    1ull << 48, 1ull << 49, 1ull << 50, 1ull << 51, 1ull << 52, 1ull << 53,
    1ull << 54, 1ull << 55, 1ull << 56, 1ull << 57, 1ull << 58, 1ull << 59,
    1ull << 60, 1ull << 61, 1ull << 62, 1ull << 63};

inline std::vector<size_t> bit_pos(uint64_t val, size_t mul) {
  std::vector<size_t> retval;
  for (size_t outer = 0; outer < 64; outer += 8) {
    size_t base = 64 * mul + outer;
    if (val & (lookup[outer])) retval.push_back(base);
    if (val & (lookup[outer + 1])) retval.push_back(base + 1);
    if (val & (lookup[outer + 2])) retval.push_back(base + 2);
    if (val & (lookup[outer + 3])) retval.push_back(base + 3);
    if (val & (lookup[outer + 4])) retval.push_back(base + 4);
    if (val & (lookup[outer + 5])) retval.push_back(base + 5);
    if (val & (lookup[outer + 6])) retval.push_back(base + 6);
    if (val & (lookup[outer + 7])) retval.push_back(base + 7);
  }
  return retval;
}
}  // namespace GDBSCAN::helper

#endif  // GDBSCAN_INCLUDE_HELPER_H_
