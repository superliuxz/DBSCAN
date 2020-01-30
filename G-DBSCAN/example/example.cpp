#include <iostream>

#include "../include/Solver.h"
#include "../include/Distance.h"

int main(int argc, char *argv[]) {
  (void) argc;

  auto logger = spdlog::stdout_color_mt("console");
  logger->set_level(spdlog::level::info);

  auto runner =
      GDBSCAN::make_solver<GDBSCAN::distance::EuclideanTwoD>(std::string(argv[1]),
                                                             10,
                                                             0.3);
  runner->prepare_dataset();
  runner->make_graph();
  runner->identify_cluster();

  return 0;
}
