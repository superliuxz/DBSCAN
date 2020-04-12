//
// Created by will on 2020-03-23.
//

#include <chrono>
#include <cxxopts.hpp>
#include <fstream>

#include "gdbscan.cuh"

int main(int argc, char *argv[]) {
#if defined(DBSCAN_TESTING)
  fprintf(stderr, "DBSCAN_TESTING enabled, something is wrong...\n");
  return 0;
#endif
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

  auto t0 = high_resolution_clock::now();
  auto d0 = duration_cast<duration<double>>(t0 - start);
  solver.sort_input_by_l1norm();
  printf("sort_input_by_l1norm takes %lf seconds\n", d0.count());

  solver.calc_num_neighbours();
  auto t1 = high_resolution_clock::now();
  auto d1 = duration_cast<duration<double>>(t1 - t0);
  printf("calc_num_neighbours takes %lf seconds\n", d1.count());

  solver.calc_start_pos();
  auto t2 = high_resolution_clock::now();
  auto d2 = duration_cast<duration<double>>(t2 - t1);
  printf("calc_start_pos takes %lf seconds\n", d2.count());

  solver.append_neighbours();
  auto t3 = high_resolution_clock::now();
  auto d3 = duration_cast<duration<double>>(t3 - t2);
  printf("append_neighbours takes %lf seconds\n", d3.count());

  solver.identify_cores();
  auto t4 = high_resolution_clock::now();
  auto d4 = duration_cast<duration<double>>(t4 - t3);
  printf("identify_cores takes %lf seconds\n", d4.count());

  solver.identify_clusters();
  auto t5 = high_resolution_clock::now();
  auto d5 = duration_cast<duration<double>>(t5 - t4);
  printf("identify_clusters takes %lf seconds\n", d5.count());

  duration<double> time_spent =
      duration_cast<duration<double>>(high_resolution_clock::now() - start);
  printf("GDBSCAN takes %lf seconds\n", time_spent.count());

  if (output_labels) {
    std::cout << "cluster ids:" << std::endl;
    for (const auto &id : solver.cluster_ids) std::cout << id << std::endl;
  }

  return 0;
}