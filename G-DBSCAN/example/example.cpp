#include <iostream>

#include "../include/Solver.h"
#include "../include/Distance.h"

int main(int argc, char *argv[]) {
  (void) argc;

  auto logger = spdlog::stdout_color_mt("console");
  logger->set_level(spdlog::level::info);

  uint min_pts = std::stoi(argv[2]);
  float radius = std::stof(argv[3]);

  auto runner =
      GDBSCAN::make_solver<GDBSCAN::distance::EuclideanTwoD>(std::string(argv[1]),
                                                             min_pts,
                                                             radius);
  runner->prepare_dataset();
  runner->make_graph();
  runner->identify_cluster();

  return 0;
}
