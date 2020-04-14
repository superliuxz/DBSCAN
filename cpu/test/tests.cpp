//
// Created by William Liu on 2020-01-23.
//

#include <gmock/gmock.h>  // ASSERT_THAT, testing::ElementsAre
#include <gtest/gtest.h>

#include "graph.h"
#include "solver.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace DBSCAN_TestVariables {
std::string abs_loc;
}

class GDBSCAN_TestEnvironment : public testing::Environment {
 public:
  explicit GDBSCAN_TestEnvironment(const std::string& command_line_arg) {
    DBSCAN_TestVariables::abs_loc = command_line_arg;
  }
};

TEST(Graph, ctor_success) {
  DBSCAN::Graph g(5, 1);
  EXPECT_EQ(g.Va.size(), 10);
  EXPECT_TRUE(g.Ea.empty());
}

TEST(Graph, insert_edge_success) {
  DBSCAN::Graph g(5, 1);
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
  DBSCAN::Graph g(5, 1);
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
  DBSCAN::Graph g(5, 1);
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
  DBSCAN::Graph g(5, 1);
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
  DBSCAN::Graph g(5, 1);
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
  DBSCAN::Graph g(5, 1);
  ASSERT_NO_THROW(g.finalize());
  ASSERT_THAT(g.Va, testing::ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  ASSERT_TRUE(g.Ea.empty());
}

TEST(TwoDimPoints, distance_squared) {
  using namespace DBSCAN::input_type;
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
  using namespace DBSCAN;
  Solver solver(DBSCAN_TestVariables::abs_loc + "/test_input1.txt", 2, 3.0f,
                1u);
  EXPECT_THAT(solver.dataset_->d1,
              testing::ElementsAre(1.0, 2.0, 2.0, 8.0, 8.0, 25.0));
  EXPECT_THAT(solver.dataset_->d2,
              testing::ElementsAre(2.0, 2.0, 3.0, 7.0, 8.0, 80.0));
}

TEST(Solver, make_graph_small_graph) {
  using namespace DBSCAN;
  Solver solver(DBSCAN_TestVariables::abs_loc + "/test_input1.txt", 2, 3.0f,
                1u);
#if !defined(BIT_ADJ)
  ASSERT_NO_THROW(solver.construct_grid());
#endif
  ASSERT_NO_THROW(solver.insert_edges());
  ASSERT_NO_THROW(solver.finalize_graph());
  ASSERT_NO_THROW(solver.classify_vertices());
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
   * even though in Va, vertex 5's neighbours starts at index 8 in Ea, but since
   * vertex 5 has not neighbours, so Ea does not actually have index 8.
   */
  EXPECT_THAT(solver.graph_->Va,
              testing::ElementsAre(0, 2, 2, 2, 4, 2, 6, 1, 7, 1, 8, 0));
  EXPECT_THAT(solver.graph_->Ea, testing::ElementsAre(1, 2, 0, 2, 0, 1, 4, 3));
  EXPECT_THAT(solver.memberships,
              testing::ElementsAre(Core, Core, Core, Noise, Noise, Noise));
}

TEST(Solver, test_input1) {
  using namespace DBSCAN;
  Solver solver(DBSCAN_TestVariables::abs_loc + "/test_input1.txt", 2, 3.0f,
                1u);
#if !defined(BIT_ADJ)
  ASSERT_NO_THROW(solver.construct_grid());
#endif
  ASSERT_NO_THROW(solver.insert_edges());
  ASSERT_NO_THROW(solver.finalize_graph());
  ASSERT_NO_THROW(solver.classify_vertices());
  ASSERT_NO_THROW(solver.identify_cluster());
  // vertices 0 1 and 2 are core vertices with cluster id = 0; vertices 3 4 and
  // 5 are noise vertices hence cluster id = -1.
  EXPECT_THAT(solver.cluster_ids, testing::ElementsAre(0, 0, 0, -1, -1, -1));
}

TEST(Solver, test_input2) {
  using namespace DBSCAN;
  Solver solver(DBSCAN_TestVariables::abs_loc + "/test_input2.txt", 2, 3.0f,
                1u);
#if !defined(BIT_ADJ)
  ASSERT_NO_THROW(solver.construct_grid());
#endif
  ASSERT_NO_THROW(solver.insert_edges());
  ASSERT_NO_THROW(solver.finalize_graph());
  ASSERT_NO_THROW(solver.classify_vertices());
  ASSERT_NO_THROW(solver.identify_cluster());
  EXPECT_THAT(solver.memberships,
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
  EXPECT_THAT(solver.cluster_ids,
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
  using namespace DBSCAN;
  Solver solver(DBSCAN_TestVariables::abs_loc + "/test_input3.txt", 3, 3.0f,
                1u);
#if !defined(BIT_ADJ)
  ASSERT_NO_THROW(solver.construct_grid());
#endif
  ASSERT_NO_THROW(solver.insert_edges());
  ASSERT_NO_THROW(solver.finalize_graph());
  ASSERT_NO_THROW(solver.classify_vertices());
  ASSERT_NO_THROW(solver.identify_cluster());
  EXPECT_THAT(solver.memberships,
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
  EXPECT_THAT(solver.cluster_ids,
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

TEST(Solver, test_input_20k) {
  using namespace DBSCAN;
  Solver solver(DBSCAN_TestVariables::abs_loc + "/test_input_20k.txt", 30,
                0.15f, 1u);
#if !defined(BIT_ADJ)
  ASSERT_NO_THROW(solver.construct_grid());
#endif
  ASSERT_NO_THROW(solver.insert_edges());
  ASSERT_NO_THROW(solver.finalize_graph());
  ASSERT_NO_THROW(solver.classify_vertices());
  ASSERT_NO_THROW(solver.identify_cluster());
  std::vector<int> expected_labels;
  std::ifstream ifs(DBSCAN_TestVariables::abs_loc +
                    "/test_input_20k_labels.txt");
  int label;
  while (ifs >> label) expected_labels.push_back(label);
  EXPECT_THAT(solver.cluster_ids, testing::ElementsAreArray(expected_labels));
}

// TODO: this test _could_ fail because DBSCAN result depends on the order of
// visiting. In the multi-threaded context, the order is nondeterministic. For
// now, run multiple times until pass. I believe some carefully picked
// clustering parameters could be robust to the nondeterministic behavior.
TEST(Solver, test_input_20k_four_threads) {
  using namespace DBSCAN;
  Solver solver(DBSCAN_TestVariables::abs_loc + "/test_input_20k.txt", 30,
                0.15f, 4u);
#if !defined(BIT_ADJ)
  ASSERT_NO_THROW(solver.construct_grid());
#endif

      ASSERT_NO_THROW(solver.insert_edges());
  ASSERT_NO_THROW(solver.finalize_graph());
  ASSERT_NO_THROW(solver.classify_vertices());
  ASSERT_NO_THROW(solver.identify_cluster());
  std::vector<int> expected_labels;
  std::ifstream ifs(DBSCAN_TestVariables::abs_loc +
                    "/test_input_20k_labels.txt");
  int label;
  while (ifs >> label) expected_labels.push_back(label);
  EXPECT_THAT(solver.cluster_ids, testing::ElementsAreArray(expected_labels));
}

int main(int argc, char* argv[]) {
  auto logger = spdlog::stdout_color_mt("console");
  logger->set_level(spdlog::level::off);
  testing::InitGoogleTest(&argc, argv);
  testing::AddGlobalTestEnvironment(new GDBSCAN_TestEnvironment(argv[1]));
  return RUN_ALL_TESTS();
}
