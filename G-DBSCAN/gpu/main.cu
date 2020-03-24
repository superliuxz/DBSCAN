//
// Created by will on 2020-03-23.
//
#include <cxxopts.hpp>
#include <fstream>

#include "gdbscan.cuh"

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

  float xs[num_nodes], ys[num_nodes];
  uint64_t num_neighbours[num_nodes], start_pos[num_nodes];

  uint64_t n;
  float x, y;
  while (ifs >> n >> x >> y) {
    xs[n] = x;
    ys[n] = y;
  }

  GDBSCAN::calc_num_neighbours(xs, ys, num_neighbours, radius * radius,
                               num_nodes);
  if (output_labels) {
    std::cout << "num_neighbours:" << std::endl;
    for (auto i = 0u; i < num_nodes; ++i)
      std::cout << i << " " << num_neighbours[i] << std::endl;
  }

  GDBSCAN::calc_start_pos(num_neighbours, start_pos, num_nodes);
  if (output_labels) {
    std::cout << "start_pos:" << std::endl;
    for (auto i = 0u; i < num_nodes; ++i)
      std::cout << i << " " << start_pos[i] << std::endl;
  }

  const uint64_t nbarr_sz =
      start_pos[num_nodes - 1] + num_neighbours[num_nodes - 1];
  std::cout << "size of neighbours array: " << nbarr_sz << std::endl;
  auto *neighbours = new uint64_t[nbarr_sz];
//  uint64_t neighbours[nbarr_sz];
  std::cout << "huhuhuhuh" << std::endl;
  GDBSCAN::append_neighbours(xs, ys, start_pos, neighbours, num_nodes,
                             nbarr_sz, radius * radius);
  if (output_labels) {
    std::cout << "neighbours:" << std::endl;
    for (auto i = 0u; i < nbarr_sz; ++i)
      std::cout << i << " " << neighbours[i] << std::endl;
  }
  delete []neighbours;
  return 0;
}