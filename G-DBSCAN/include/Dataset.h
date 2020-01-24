//
// Created by William Liu on 2020-01-24.
//

#ifndef GDBSCAN_INCLUDE_DATASET_H_
#define GDBSCAN_INCLUDE_DATASET_H_

#include <sstream>
#include <vector>

namespace GDBSCAN {

template<class DimensionType>
class Dataset {
 public:
  explicit Dataset(size_t num_nodes) : size_(num_nodes),
                                       in_dataset_(num_nodes, true),
                                       positions_(num_nodes, DimensionType()) {}
  void exclude(size_t node) {
    check_oob(node);
    in_dataset_[node] = false;
  }
  bool in_dataset(size_t node) const {
    check_oob(node);
    return in_dataset_[node];
  }
  // setter
  DimensionType &operator[](const size_t &node) {
    check_oob(node);
    in_dataset_[node] = true;
    return positions_[node];
  }
  // getter
  const DimensionType &operator[](const size_t &node) const {
    check_oob(node);
    return positions_[node];
  }
 private:
  size_t size_;
  std::vector<bool> in_dataset_;
  std::vector<DimensionType> positions_;
  void check_oob(const size_t &pos) const {
    if (pos < 0 || pos >= size_) {
      std::ostringstream oss;
      oss << pos << " out of bound!" << std::endl;
      throw std::runtime_error(oss.str());
    }
  }
};
} // namespace GDBSCAN

#endif //GDBSCAN_INCLUDE_DATASET_H_
