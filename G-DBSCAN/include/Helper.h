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
inline size_t popcount_bool(const std::vector<bool>& vec) {
  std::vector<size_t> temp(8, 0llu);
  for (auto i = 0llu; i < vec.size(); i += 8) {
    temp[0] += vec[i] != 0;
    temp[1] += vec[i + 1] != 0;
    temp[2] += vec[i + 2] != 0;
    temp[3] += vec[i + 3] != 0;
    temp[4] += vec[i + 4] != 0;
    temp[5] += vec[i + 5] != 0;
    temp[6] += vec[i + 6] != 0;
    temp[7] += vec[i + 7] != 0;
  }
  return std::accumulate(temp.cbegin(), temp.cend(), 0llu);
}
// clang-format off
// see https://en.wikipedia.org/wiki/Hamming_weight#Efficient_implementation
const uint64_t m1  = 0x5555555555555555; //binary: 0101...
const uint64_t m2  = 0x3333333333333333; //binary: 00110011..
const uint64_t m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
const uint64_t h01 = 0x0101010101010101; //the sum of 256 to the power of 0,1,2,3...
inline size_t popcount64(uint64_t x) {
  x -= (x >> 1u) & m1;              // put count of each 2 bits into those 2 bits
  x = (x & m2) + ((x >> 2u) & m2);  // put count of each 4 bits into those 4 bits
  x = (x + (x >> 4u)) & m4;         // put count of each 8 bits into those 8 bits
  return (x * h01) >> 56u;  // returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ...
}
// clang-format on
const uint64_t lookup[64] = {
    1ull << 0u,  1ull << 1u,  1ull << 2u,  1ull << 3u,  1ull << 4u,
    1ull << 5u,  1ull << 6u,  1ull << 7u,  1ull << 8u,  1ull << 9u,
    1ull << 10u, 1ull << 11u, 1ull << 12u, 1ull << 13u, 1ull << 14u,
    1ull << 15u, 1ull << 16u, 1ull << 17u, 1ull << 18u, 1ull << 19u,
    1ull << 20u, 1ull << 21u, 1ull << 22u, 1ull << 23u, 1ull << 24u,
    1ull << 25u, 1ull << 26u, 1ull << 27u, 1ull << 28u, 1ull << 29u,
    1ull << 30u, 1ull << 31u, 1ull << 32u, 1ull << 33u, 1ull << 34u,
    1ull << 35u, 1ull << 36u, 1ull << 37u, 1ull << 38u, 1ull << 39u,
    1ull << 40u, 1ull << 41u, 1ull << 42u, 1ull << 43u, 1ull << 44u,
    1ull << 45u, 1ull << 46u, 1ull << 47u, 1ull << 48u, 1ull << 49u,
    1ull << 50u, 1ull << 51u, 1ull << 52u, 1ull << 53u, 1ull << 54u,
    1ull << 55u, 1ull << 56u, 1ull << 57u, 1ull << 58u, 1ull << 59u,
    1ull << 60u, 1ull << 61u, 1ull << 62u, 1ull << 63u};

inline std::vector<size_t> bit_pos(uint64_t val, size_t mul) {
  std::vector<size_t> retval;
  retval.reserve(64u);  // set the capacity to 64.
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

inline std::vector<size_t> true_pos(const std::vector<bool>& nbs) {
  std::vector<size_t> retval;
  for (size_t i = 0llu; i < nbs.size(); i += 8) {
    if (nbs[i]) retval.push_back(i);
    if (nbs[i + 1]) retval.push_back(i + 1);
    if (nbs[i + 2]) retval.push_back(i + 2);
    if (nbs[i + 3]) retval.push_back(i + 3);
    if (nbs[i + 4]) retval.push_back(i + 4);
    if (nbs[i + 5]) retval.push_back(i + 5);
    if (nbs[i + 6]) retval.push_back(i + 6);
    if (nbs[i + 7]) retval.push_back(i + 7);
  }
  return retval;
}
}  // namespace GDBSCAN::helper

#endif  // GDBSCAN_INCLUDE_HELPER_H_
