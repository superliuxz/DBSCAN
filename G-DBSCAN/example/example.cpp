#include <iostream>

#include "../include/Solver.h"
#include "../include/Dimension.h"

int main(int argc, char *argv[]) {
  auto logger = spdlog::stdout_color_mt("console");
  logger->set_level(spdlog::level::info);

  auto runner =
      GDBSCAN::make_solver<GDBSCAN::dimension::TwoD>(std::string(argv[1]),
                                                     10,
                                                     0.3);
  runner->prepare_dataset();
  runner->make_graph();
  runner->identify_cluster();

  auto graph = runner->graph_view();

  for (auto &v: graph.cluster_ids) {
    std::cout << v << std::endl;
  }

  return 0;
}
