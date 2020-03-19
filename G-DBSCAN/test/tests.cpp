//
// Created by William Liu on 2020-01-23.
//

#include <gmock/gmock.h>  // ASSERT_THAT, testing::ElementsAre
#include <gtest/gtest.h>

#include "../src/graph.h"
#include "../src/solver.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace GDBSCAN_TestVariables {
std::string abs_loc;
}

class GDBSCAN_TestEnvironment : public testing::Environment {
 public:
  explicit GDBSCAN_TestEnvironment(const std::string& command_line_arg) {
    GDBSCAN_TestVariables::abs_loc = command_line_arg;
  }
};

TEST(Graph, ctor_success) {
  GDBSCAN::Graph g(5, 1);
  EXPECT_EQ(g.Va.size(), 10);
  EXPECT_EQ(g.cluster_ids.size(), 5);
  EXPECT_TRUE(g.Ea.empty());
}

TEST(Graph, insert_edge_success) {
  GDBSCAN::Graph g(5, 1);
#if defined(BIT_ADJ)
  ASSERT_NO_THROW(g.insert_edge(2, 0, 1u << 1u));
  ASSERT_NO_THROW(g.insert_edge(2, 0, 1u << 4u));
  ASSERT_NO_THROW(g.insert_edge(2, 0, 1u << 0u));
  ASSERT_NO_THROW(g.insert_edge(0, 0, 1u << 3u));
#else
  ASSERT_NO_THROW(g.insert_edge(2, 1));
  ASSERT_NO_THROW(g.insert_edge(2, 4));
  ASSERT_NO_THROW(g.insert_edge(2, 0));
  ASSERT_NO_THROW(g.insert_edge(0, 3));
#endif
}

TEST(Graph, insert_edge_failed_oob) {
  GDBSCAN::Graph g(5, 1);
#if defined(BIT_ADJ)
  ASSERT_NO_THROW(g.insert_edge(2, 0, 1u << 1u));
  ASSERT_THROW(g.insert_edge(0, 1, 1u << 5u), std::runtime_error);
  ASSERT_THROW(g.insert_edge(-1, 0, 1u << 2u), std::runtime_error);
  ASSERT_THROW(g.insert_edge(-2, 0, 1u << 9u), std::runtime_error);
#else
  ASSERT_NO_THROW(g.insert_edge(2, 1));
  ASSERT_THROW(g.insert_edge(0, 5), std::runtime_error);
  ASSERT_THROW(g.insert_edge(-1, 2), std::runtime_error);
  ASSERT_THROW(g.insert_edge(-2, 9), std::runtime_error);
#endif
}

TEST(Graph, finalize_success) {
  GDBSCAN::Graph g(5, 1);
#if defined(BIT_ADJ)
  ASSERT_NO_THROW(g.insert_edge(2, 0, 1u << 1u));
  ASSERT_NO_THROW(g.insert_edge(1, 0, 1u << 2u));
  ASSERT_NO_THROW(g.insert_edge(2, 0, 1u << 4u));
  ASSERT_NO_THROW(g.insert_edge(4, 0, 1u << 2u));
  ASSERT_NO_THROW(g.insert_edge(2, 0, 1u << 0u));
  ASSERT_NO_THROW(g.insert_edge(0, 0, 1u << 2u));
  ASSERT_NO_THROW(g.insert_edge(0, 0, 1u << 3u));
  ASSERT_NO_THROW(g.insert_edge(3, 0, 1u << 0u));
#else
  ASSERT_NO_THROW(g.insert_edge(2, 1));
  ASSERT_NO_THROW(g.insert_edge(1, 2));
  ASSERT_NO_THROW(g.insert_edge(2, 4));
  ASSERT_NO_THROW(g.insert_edge(4, 2));
  ASSERT_NO_THROW(g.insert_edge(2, 0));
  ASSERT_NO_THROW(g.insert_edge(0, 2));
  ASSERT_NO_THROW(g.insert_edge(0, 3));
  ASSERT_NO_THROW(g.insert_edge(3, 0));
#endif
  ASSERT_NO_THROW(g.finalize());

  ASSERT_THAT(g.Va, testing::ElementsAre(0, 2, 2, 1, 3, 3, 6, 1, 7, 1));
// if use bit adjacency matrix, the neighbours are ascending order, where
// as other type of adjacency list respect the insertion order.
#if defined(BIT_ADJ)
  ASSERT_THAT(g.Ea, testing::ElementsAre(2, 3, 2, 0, 1, 4, 0, 2));
#else
  ASSERT_THAT(g.Ea, testing::ElementsAre(2, 3, 2, 1, 4, 0, 0, 2));
#endif
}

TEST(Graph, finalize_fail_second_finalize) {
  GDBSCAN::Graph g(5, 1);
#if defined(BIT_ADJ)
  ASSERT_NO_THROW(g.insert_edge(2, 0, 1u << 1u));
  ASSERT_NO_THROW(g.insert_edge(2, 0, 1u << 4u));
  ASSERT_NO_THROW(g.insert_edge(2, 0, 1u << 0u));
  ASSERT_NO_THROW(g.insert_edge(0, 0, 1u << 3u));
#else
  ASSERT_NO_THROW(g.insert_edge(2, 1));
  ASSERT_NO_THROW(g.insert_edge(2, 4));
  ASSERT_NO_THROW(g.insert_edge(2, 0));
  ASSERT_NO_THROW(g.insert_edge(0, 3));
#endif
  ASSERT_NO_THROW(g.finalize());

  ASSERT_THROW(g.finalize(), std::runtime_error);
}

TEST(Graph, finalize_success_disconnected_graph) {
  GDBSCAN::Graph g(5, 1);
#if defined(BIT_ADJ)
  ASSERT_NO_THROW(g.insert_edge(2, 0, 1u << 1u));
  ASSERT_NO_THROW(g.insert_edge(1, 0, 1u << 2u));
  ASSERT_NO_THROW(g.insert_edge(2, 0, 1u << 4u));
  ASSERT_NO_THROW(g.insert_edge(4, 0, 1u << 2u));
  ASSERT_NO_THROW(g.insert_edge(0, 0, 1u << 2u));
  ASSERT_NO_THROW(g.insert_edge(2, 0, 1u << 0u));
#else
  ASSERT_NO_THROW(g.insert_edge(2, 1));
  ASSERT_NO_THROW(g.insert_edge(1, 2));
  ASSERT_NO_THROW(g.insert_edge(2, 4));
  ASSERT_NO_THROW(g.insert_edge(4, 2));
  ASSERT_NO_THROW(g.insert_edge(0, 2));
  ASSERT_NO_THROW(g.insert_edge(2, 0));
#endif
  ASSERT_NO_THROW(g.finalize());
  ASSERT_THAT(g.Va, testing::ElementsAre(0, 1, 1, 1, 2, 3, 5, 0, 5, 1));
#if defined(BIT_ADJ)
  ASSERT_THAT(g.Ea, testing::ElementsAre(2, 2, 0, 1, 4, 2));
#else
  ASSERT_THAT(g.Ea, testing::ElementsAre(2, 2, 1, 4, 0, 2));
#endif
}

TEST(Graph, finalize_success_no_edges) {
  GDBSCAN::Graph g(5, 1);
  ASSERT_NO_THROW(g.finalize());
  ASSERT_THAT(g.Va, testing::ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  ASSERT_TRUE(g.Ea.empty());
}

TEST(Graph, classify_node_success) {
  GDBSCAN::Graph g(5, 1);
  ASSERT_NO_THROW(g.finalize());
  ASSERT_NO_THROW(g.cluster_node(0, 2));
  ASSERT_NO_THROW(g.cluster_node(1, 2));
  ASSERT_NO_THROW(g.cluster_node(2, 4));
}

TEST(Graph, classify_node_fail_no_finalize) {
  GDBSCAN::Graph g(5, 1);
  ASSERT_THROW(g.cluster_node(0, 2), std::runtime_error);
}

TEST(Graph, classify_node_fail_oob) {
  GDBSCAN::Graph g(5, 1);
  ASSERT_NO_THROW(g.finalize());
  ASSERT_NO_THROW(g.cluster_node(0, 2));
  ASSERT_THROW(g.cluster_node(-1, 2), std::runtime_error);
  ASSERT_THROW(g.cluster_node(6, 2), std::runtime_error);
}

TEST(TwoDimPoints, distance_squared) {
  using namespace GDBSCAN::input_type;
  EXPECT_FLOAT_EQ(TwoDimPoints::euclidean_distance_square(1, 2, 3, 4),
                  std::pow(1 - 3, 2) + std::pow(2 - 4, 2));
  EXPECT_FLOAT_EQ(TwoDimPoints::euclidean_distance_square(-1, -4, 4, 1),
                  std::pow(-1 - 4, 2) + std::pow(-4 - 1, 2));
  EXPECT_FLOAT_EQ(TwoDimPoints::euclidean_distance_square(0, 0, 4, 1),
                  std::pow(0 - 4, 2) + std::pow(0 - 1, 2));
  EXPECT_FLOAT_EQ(TwoDimPoints::euclidean_distance_square(1.0, 2.0, 2.5, 3.4),
                  std::pow(1.0f - 2.5f, 2) + std::pow(2.0f - 3.4f, 2));
}

TEST(Solver, prepare_dataset) {
  using namespace GDBSCAN;
  Solver<input_type::TwoDimPoints> solver(
      GDBSCAN_TestVariables::abs_loc + "/test_input1.txt", 2, 3.0f, 1u);
  auto& dataset = solver.dataset_view();
  EXPECT_THAT(dataset.d1, testing::ElementsAre(1.0, 2.0, 2.0, 8.0, 8.0, 25.0));
  EXPECT_THAT(dataset.d2, testing::ElementsAre(2.0, 2.0, 3.0, 7.0, 8.0, 80.0));
}

TEST(Solver, make_graph_small_graph) {
  using namespace GDBSCAN;
  Solver<input_type::TwoDimPoints> solver(
      GDBSCAN_TestVariables::abs_loc + "/test_input1.txt", 2, 3.0f, 1u);
  ASSERT_NO_THROW(solver.insert_edges());
  ASSERT_NO_THROW(solver.finalize_graph());
  ASSERT_NO_THROW(solver.classify_nodes());
  auto& graph = solver.graph_view();
  /*
   * Va:
   * 0 2 4 6 7 8 <- start pos in Ea
   * 2 2 2 1 1 0 <- number of neighbours
   * 0 1 2 3 4 5 <- index
   *
   * Ea:
   * 1 2 0 2 0 1 4 3 <- neighbours
   * 0 1 2 3 4 5 6 7 <- index
   *
   * even though in Va, node 5's neighbours starts at index 8 in Ea, but since
   * node 5 has not neighbours, so Ea does not actually have index 8.
   */
  EXPECT_THAT(graph.Va,
              testing::ElementsAre(0, 2, 2, 2, 4, 2, 6, 1, 7, 1, 8, 0));
  EXPECT_THAT(graph.Ea, testing::ElementsAre(1, 2, 0, 2, 0, 1, 4, 3));
  EXPECT_THAT(graph.memberships,
              testing::ElementsAre(Core, Core, Core, Noise, Noise, Noise));
}

TEST(Solver, test_input1) {
  using namespace GDBSCAN;
  Solver<input_type::TwoDimPoints> solver(
      GDBSCAN_TestVariables::abs_loc + "/test_input1.txt", 2, 3.0f, 1u);
  ASSERT_NO_THROW(solver.insert_edges());
  ASSERT_NO_THROW(solver.finalize_graph());
  ASSERT_NO_THROW(solver.classify_nodes());
  ASSERT_NO_THROW(solver.identify_cluster());
  auto& graph = solver.graph_view();
  // nodes 0 1 and 2 are core nodes with cluster id = 1; nodes 3 4 and 5 are
  // noise nodes hence cluster id = -1.
  EXPECT_THAT(graph.cluster_ids, testing::ElementsAre(0, 0, 0, -1, -1, -1));
}

TEST(Solver, test_input2) {
  using namespace GDBSCAN;
  Solver<input_type::TwoDimPoints> solver(
      GDBSCAN_TestVariables::abs_loc + "/test_input2.txt", 2, 3.0f, 1u);
  ASSERT_NO_THROW(solver.insert_edges());
  ASSERT_NO_THROW(solver.finalize_graph());
  ASSERT_NO_THROW(solver.classify_nodes());
  ASSERT_NO_THROW(solver.identify_cluster());
  auto& graph = solver.graph_view();
  EXPECT_THAT(graph.memberships,
              testing::ElementsAre(Core,    // 0
                                   Core,    // 1
                                   Core,    // 2
                                   Border,  // 3
                                   Border,  // 4
                                   Core,    // 5
                                   Core,    // 6
                                   Core,    // 7
                                   Core,    // 8
                                   Noise    // 9
                                   ));
  EXPECT_THAT(graph.cluster_ids,
              testing::ElementsAre(0,  // 0
                                   0,  // 1
                                   0,  // 2
                                   0,  // 3
                                   1,  // 4
                                   1,  // 5
                                   1,  // 6
                                   1,  // 7
                                   1,  // 8
                                   -1  // 9
                                   ));
}

TEST(Solver, test_input3) {
  using namespace GDBSCAN;
  Solver<input_type::TwoDimPoints> solver(
      GDBSCAN_TestVariables::abs_loc + "/test_input3.txt", 3, 3.0f, 1u);
  ASSERT_NO_THROW(solver.insert_edges());
  ASSERT_NO_THROW(solver.finalize_graph());
  ASSERT_NO_THROW(solver.classify_nodes());
  ASSERT_NO_THROW(solver.identify_cluster());
  auto& graph = solver.graph_view();
  EXPECT_THAT(graph.memberships,
              testing::ElementsAre(Core,    // 0
                                   Core,    // 1
                                   Core,    // 2
                                   Core,    // 3
                                   Border,  // 4
                                   Noise,   // 5
                                   Border,  // 6
                                   Core,    // 7
                                   Core,    // 8
                                   Core,    // 9
                                   Core     // 10
                                   ));
  EXPECT_THAT(graph.cluster_ids,
              testing::ElementsAre(0,   // 0
                                   0,   // 1
                                   0,   // 2
                                   0,   // 3
                                   0,   // 4
                                   -1,  // 5
                                   1,   // 6
                                   1,   // 7
                                   1,   // 8
                                   1,   // 9
                                   1    // 10
                                   ));
}

TEST(Solver, test_input4) {
  using namespace GDBSCAN;
  Solver<input_type::TwoDimPoints> solver(
      GDBSCAN_TestVariables::abs_loc + "/test_input4.txt", 30, 0.15f, 1u);
  ASSERT_NO_THROW(solver.insert_edges());
  ASSERT_NO_THROW(solver.finalize_graph());
  ASSERT_NO_THROW(solver.classify_nodes());
  ASSERT_NO_THROW(solver.identify_cluster());
  std::vector<int> expected_labels;
  std::ifstream ifs(GDBSCAN_TestVariables::abs_loc + "/test_input4_labels.txt");
  int label;
  while (ifs >> label) expected_labels.push_back(label);
  auto& graph = solver.graph_view();
  EXPECT_THAT(graph.cluster_ids, testing::ElementsAreArray(expected_labels));
}

// TODO: this test _could_ fail because DBSCAN result depends on the order of
// visiting. In the multi-threaded context, the order is nondeterministic. For
// now, run multiple times until pass. I believe some carefully picked
// clustering parameters could be robust to the nondeterministic behavior.
TEST(Solver, test_input4_four_threads) {
  using namespace GDBSCAN;
  Solver<input_type::TwoDimPoints> solver(
      GDBSCAN_TestVariables::abs_loc + "/test_input4.txt", 30, 0.15f, 4u);
  ASSERT_NO_THROW(solver.insert_edges());
  ASSERT_NO_THROW(solver.finalize_graph());
  ASSERT_NO_THROW(solver.classify_nodes());
  ASSERT_NO_THROW(solver.identify_cluster());
  std::vector<int> expected_labels;
  std::ifstream ifs(GDBSCAN_TestVariables::abs_loc + "/test_input4_labels.txt");
  int label;
  while (ifs >> label) expected_labels.push_back(label);
  auto& graph = solver.graph_view();
  EXPECT_THAT(graph.cluster_ids, testing::ElementsAreArray(expected_labels));
}

int main(int argc, char* argv[]) {
  auto logger = spdlog::stdout_color_mt("console");
  logger->set_level(spdlog::level::off);
  testing::InitGoogleTest(&argc, argv);
  testing::AddGlobalTestEnvironment(new GDBSCAN_TestEnvironment(argv[1]));
  return RUN_ALL_TESTS();
}
