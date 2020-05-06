#include <spdlog/sinks/stdout_color_sinks.h>

#include <cxxopts.hpp>
#include <iostream>

#include "solver.h"

int main(int argc, char* argv[]) {
#if defined(DBSCAN_TESTING)
  fprintf(stderr, "DBSCAN_TESTING enabled, something is wrong...\n");
  return 0;
#endif
  auto logger = spdlog::stdout_color_mt("console");
  logger->set_level(spdlog::level::info);

  cxxopts::Options options("DBSCAN", "ma, look, it's DBSCAN");
  // clang-format off
  options.add_options()
      ("p,print", "Print clustering IDs") // boolean
      ("r,eps", "Clustering radius", cxxopts::value<float>())
      ("n,min-pts", "Number of points within radius", cxxopts::value<size_t>())
      ("i,input", "Input filename", cxxopts::value<std::string>())
      ("t,num-threads", "Number of threads", cxxopts::value<uint8_t>()->default_value("1"))
      ;
  // clang-format on
  auto args = options.parse(argc, argv);

  bool output_labels = args["print"].as<bool>();
  float radius = args["eps"].as<float>();
  uint min_pts = args["min-pts"].as<size_t>();
  std::string input = args["input"].as<std::string>();
  uint8_t num_threads = args["num-threads"].as<uint8_t>();

  logger->debug("radius {} min_pts {}", radius, min_pts);

  DBSCAN::Solver solver(input, min_pts, radius, num_threads);
  auto const start = std::chrono::high_resolution_clock::now();
#if !defined(BIT_ADJ)
  solver.construct_grid();
#endif
  solver.InsertEdges();
  solver.FinalizeGraph();
  solver.ClassifyNoises();
  solver.IdentifyClusters();
  auto const end = std::chrono::high_resolution_clock::now();
  auto const duration =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  spdlog::info("DBSCAN takes {} sec", duration.count());

  if (output_labels) {
    for (const auto& l : solver.cluster_ids) {
      std::cout << l << std::endl;
    }
  }

  return 0;
}
