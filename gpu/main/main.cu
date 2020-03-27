//
// Created by will on 2020-03-23.
//

#include <chrono>
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

  using namespace std::chrono;
  high_resolution_clock::time_point start = high_resolution_clock::now();

  GDBSCAN::Solver solver(input, min_pts, radius);

  solver.calc_num_neighbours();
  solver.calc_start_pos();
  solver.append_neighbours();
  solver.identify_cores();
  solver.identify_clusters();

  duration<double> time_spent =
      duration_cast<duration<double>>(high_resolution_clock::now() - start);
  std::cout << "G-DBSCAN takes " << time_spent.count() << " seconds"
            << std::endl;

  if (output_labels) {
    std::cout << "cluster ids:" << std::endl;
    for (const auto &id : solver.cluster_ids) std::cout << id << std::endl;
  }

  return 0;
}