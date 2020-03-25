//
// Created by will on 2020-03-23.
//
#include <thrust/device_vector.h>

#include <cxxopts.hpp>
#include <fstream>

#include "gdbscan.cuh"
#include "membership.h"

int main(int argc, char *argv[]) {
  cxxopts::Options options("GDBSCAN", "ma, look, it's GDBSCAN");
  // clang-format off
    options.add_options()
            ("p,print", "Print clustering IDs") // boolean
            ("r,eps", "Clustering radius", cxxopts::value<float>())
            ("n,min-samples", "Number of points within radius", cxxopts::value<size_t>())
            ("i,input", "Input filename", cxxopts::value<std::string>());
  // clang-format on
  auto args = options.parse(argc, argv);

  bool output_labels = args["print"].as<bool>();
  float radius = args["eps"].as<float>();
  uint min_pts = args["min-samples"].as<size_t>();
  std::string input = args["input"].as<std::string>();

  std::cout << "minPts=" << min_pts << "; eps=" << radius << std::endl;

  uint64_t num_nodes = 0;

  auto ifs = std::ifstream(input);
  ifs >> num_nodes;

  thrust::host_vector<float> xs(num_nodes, 0), ys(num_nodes, 0);
  thrust::host_vector<uint64_t> num_neighbours(num_nodes, 0),
      start_pos(num_nodes, 0);

  uint64_t n;
  float x, y;
  while (ifs >> n >> x >> y) {
    xs[n] = x;
    ys[n] = y;
  }

  GDBSCAN::calc_num_neighbours(xs, ys, num_neighbours, radius * radius);
  if (output_labels) {
    std::cout << "num_neighbours:" << std::endl;
    for (auto i = 0u; i < num_nodes; ++i)
      std::cout << i << " " << num_neighbours[i] << std::endl;
  }

  GDBSCAN::calc_start_pos(thrust::raw_pointer_cast(num_neighbours.data()),
                          thrust::raw_pointer_cast(start_pos.data()),
                          num_nodes);
  if (output_labels) {
    std::cout << "start_pos:" << std::endl;
    for (auto i = 0u; i < num_nodes; ++i)
      std::cout << i << " " << start_pos[i] << std::endl;
  }

  const uint64_t nbarr_sz =
      start_pos[num_nodes - 1] + num_neighbours[num_nodes - 1];
  std::cout << "size of neighbours array: " << nbarr_sz << std::endl;
  thrust::host_vector<uint64_t> neighbours(nbarr_sz, 0);

  GDBSCAN::append_neighbours(thrust::raw_pointer_cast(xs.data()),
                             thrust::raw_pointer_cast(ys.data()),
                             thrust::raw_pointer_cast(start_pos.data()),
                             thrust::raw_pointer_cast(neighbours.data()),
                             num_nodes, nbarr_sz, radius * radius);
  if (output_labels) {
    std::cout << "neighbours:" << std::endl;
    for (auto i = 0u; i < nbarr_sz; ++i)
      std::cout << i << " " << neighbours[i] << std::endl;
  }

  thrust::host_vector<DBSCAN::membership> membership(num_nodes,
                                                     DBSCAN::membership::Noise);
  for (uint64_t i = 0; i < num_nodes; ++i) {
    if (num_neighbours[i] >= min_pts) membership[i] = DBSCAN::membership::Core;
  }
  if (output_labels) {
    std::cout << "membership (Core vs non-Core):" << std::endl;
    for (auto i = 0u; i < num_nodes; ++i)
      std::cout << i << " "
                << (membership[i] == DBSCAN::membership::Core ? "Core"
                                                              : "non-Core")
                << std::endl;
  }

  return 0;
}