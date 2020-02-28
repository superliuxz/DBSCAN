#include <cxxopts.hpp>
#include <iostream>

#include "include/Point.h"
#include "include/Solver.h"

int main(int argc, char* argv[]) {
  auto logger = spdlog::stdout_color_mt("console");
  logger->set_level(spdlog::level::info);

  cxxopts::Options options("GDBSCAN", "GDBSCAN");
  // clang-format off
  options.add_options()
      ("p,print", "Output clustering IDs") // boolean
      ("r,eps", "clustering radius", cxxopts::value<float>())
      ("n,min-samples", "Number of points within radius", cxxopts::value<size_t>())
      ("i,input", "Input filename", cxxopts::value<std::string>())
      ("t,num-threads", "Number of threads", cxxopts::value<uint8_t>()->default_value("1"))
      ;
  // clang-format on
  auto args = options.parse(argc, argv);

  bool output_labels = args["print"].as<bool>();
  float radius = args["eps"].as<float>();
  uint min_pts = args["min-samples"].as<size_t>();
  std::string input = args["input"].as<std::string>();
  uint8_t num_threads = args["num-threads"].as<uint8_t>();

  logger->debug("radius {} min_pts {}", radius, min_pts);

  auto runner = GDBSCAN::make_solver<GDBSCAN::point::EuclideanTwoD>(
      std::string(input), min_pts, radius, num_threads);
  runner->insert_edges();
  runner->finalize_graph();
  runner->classify_nodes();
  runner->identify_cluster();

  if (output_labels) {
    auto g = runner->graph_view();
    for (const auto& l : g.cluster_ids) {
      std::cout << l << std::endl;
    }
  }

  return 0;
}
