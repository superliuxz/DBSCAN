#include <iostream>

#include "../include/Point.h"
#include "../include/Solver.h"

int main(int argc, char *argv[]) {
  (void)argc;

  auto logger = spdlog::stdout_color_mt("console");
  logger->set_level(spdlog::level::info);

  float radius = std::stof(argv[2]);
  uint min_pts = std::stoi(argv[3]);

  logger->debug("radius {} min_pts {}", radius, min_pts);

  auto runner = GDBSCAN::make_solver<GDBSCAN::point::EuclideanTwoD>(
      std::string(argv[1]), min_pts, radius);
  runner->prepare_dataset();
  runner->make_graph();
  runner->identify_cluster();

  return 0;
}
