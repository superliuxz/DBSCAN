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

  GDBSCAN::Solver solver(input, min_pts, radius);

  solver.calc_num_neighbours();
  if (output_labels) {
    std::cout << "num_neighbours:" << std::endl;
    for (auto i = 0u; i < solver.num_vtx_; ++i)
      std::cout << i << " " << solver.num_neighbours_[i] << std::endl;
  }

  solver.calc_start_pos();
  if (output_labels) {
    std::cout << "start_pos:" << std::endl;
    for (auto i = 0u; i < solver.num_vtx_; ++i)
      std::cout << i << " " << solver.start_pos_[i] << std::endl;
  }

  solver.append_neighbours();
  if (output_labels) {
    std::cout << "neighbours:" << std::endl;
    for (auto i = 0u; i < solver.neighbours_.size(); ++i)
      std::cout << i << " " << solver.neighbours_[i] << std::endl;
  }

  solver.identify_cores();

  solver.identify_clusters();
  if (output_labels) {
    std::cout << "membership:" << std::endl;
    for (auto i = 0u; i < solver.num_vtx_; ++i) {
      if (solver.membership_[i] == DBSCAN::membership::Core)
        std::cout << i << " Core" << std::endl;
      else if (solver.membership_[i] == DBSCAN::membership::Border)
        std::cout << i << " Border" << std::endl;
      else
        std::cout << i << " Noise" << std::endl;
    }
  }
  if (output_labels) {
    std::cout << "cluster ids:" << std::endl;
    for (auto i = 0u; i < solver.num_vtx_; ++i)
      std::cout << i << " " << solver.cluster_ids_[i] << std::endl;
  }

  return 0;
}