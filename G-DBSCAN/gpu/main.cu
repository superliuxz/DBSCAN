//
// Created by will on 2020-03-23.
//
#include <cxxopts.hpp>
#include <fstream>

#include "gdbscan.cuh"

int main(int argc, char* argv[]) {
  cxxopts::Options options("GDBSCAN", "ma, look, it's GDBSCAN");
  // clang-format off
    options.add_options()
            ("p,print", "Print clustering IDs") // boolean
            ("r,eps", "Clustering radius", cxxopts::value<float>())
            ("n,min-samples", "Number of points within radius", cxxopts::value<size_t>())
            ("i,input", "Input filename", cxxopts::value<std::string>())
            ;
  // clang-format on
  auto args = options.parse(argc, argv);

  bool output_labels = args["print"].as<bool>();
  float radius = args["eps"].as<float>();
  uint min_pts = args["min-samples"].as<size_t>();
  std::string input = args["input"].as<std::string>();

  std::cout << "minPts=" << min_pts << "; eps=" << radius << std::endl;

  uint64_t num_nodes = 0;
  float xs[num_nodes], ys[num_nodes];
  uint64_t Va[num_nodes];

  auto ifs = std::ifstream(input);
  ifs >> num_nodes;
  uint64_t n;
  float x, y;
  while (ifs >> n >> x >> y) {
    xs[n] = x;
    ys[n] = y;
  }

  GDBSCAN::insert_edge(xs, ys, Va, radius * radius, num_nodes);

  if (output_labels) {
    for (auto i = 0u; i < num_nodes; ++i)
      std::cout << i << " " << Va[i] << std::endl;
  }
  return 0;
}