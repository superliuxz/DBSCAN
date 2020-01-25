//
// Created by William Liu on 2020-01-23.
//

#include <gmock/gmock.h> // ASSERT_THAT, testing::ElementsAre
#include <gtest/gtest.h>

#include "../include/Dataset.h"
#include "../include/Dimension.h"
#include "../include/Graph.h"
#include "../include/Runner.h"

TEST(Graph, ctor_success) {
  GDBSCAN::Graph g(5);
  EXPECT_EQ(g.Va.size(), 10);
  EXPECT_EQ(g.cluster_ids.size(), 5);
  EXPECT_TRUE(g.Ea.empty());
}

TEST(Graph, insert_edge_success) {
  GDBSCAN::Graph g(5);
  ASSERT_NO_THROW(g.insert_edge(2, 1));
  ASSERT_NO_THROW(g.insert_edge(2, 4));
  ASSERT_NO_THROW(g.insert_edge(2, 0));
  ASSERT_NO_THROW(g.insert_edge(0, 3));
}

TEST(Graph, insert_edge_failed_oob) {
  GDBSCAN::Graph g(5);
  ASSERT_NO_THROW(g.insert_edge(2, 1));
  ASSERT_THROW(g.insert_edge(0, 5), std::runtime_error);
  ASSERT_THROW(g.insert_edge(-1, 2), std::runtime_error);
  ASSERT_THROW(g.insert_edge(-2, 9), std::runtime_error);
}

TEST(Graph, finalize_success) {
  GDBSCAN::Graph g(5);
  ASSERT_NO_THROW(g.insert_edge(2, 1));
  ASSERT_NO_THROW(g.insert_edge(2, 4));
  ASSERT_NO_THROW(g.insert_edge(2, 0));
  ASSERT_NO_THROW(g.insert_edge(0, 3));
  ASSERT_NO_THROW(g.finalize());

  ASSERT_THAT(g.Va, testing::ElementsAre(2, 0, 1, 2, 3, 3, 1, 6, 1, 7));
  ASSERT_THAT(g.Ea, testing::ElementsAre(2, 3, 2, 1, 4, 0, 0, 2));
}

TEST(Graph, finalize_fail_second_finalize) {
  GDBSCAN::Graph g(5);
  ASSERT_NO_THROW(g.insert_edge(2, 1));
  ASSERT_NO_THROW(g.insert_edge(2, 4));
  ASSERT_NO_THROW(g.insert_edge(2, 0));
  ASSERT_NO_THROW(g.insert_edge(0, 3));
  ASSERT_NO_THROW(g.finalize());

  ASSERT_THROW(g.finalize(), std::runtime_error);
}

TEST(Graph, finalize_success_disconnected_graph) {
  GDBSCAN::Graph g(5);
  ASSERT_NO_THROW(g.insert_edge(2, 1));
  ASSERT_NO_THROW(g.insert_edge(2, 4));
  ASSERT_NO_THROW(g.insert_edge(2, 0));
  ASSERT_NO_THROW(g.finalize());

  ASSERT_THAT(g.Va, testing::ElementsAre(1, 0, 1, 1, 3, 2, 0, 5, 1, 5));
  ASSERT_THAT(g.Ea, testing::ElementsAre(2, 2, 1, 4, 0, 2));
}

TEST(Graph, finalize_success_no_edges) {
  GDBSCAN::Graph g(5);
  ASSERT_NO_THROW(g.finalize());
  ASSERT_THAT(g.Va, testing::ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  ASSERT_TRUE(g.Ea.empty());
}

TEST(Graph, classify_node_success) {
  GDBSCAN::Graph g(5);
  ASSERT_NO_THROW(g.finalize());
  ASSERT_NO_THROW(g.cluster_node(0, 2));
  ASSERT_NO_THROW(g.cluster_node(1, 2));
  ASSERT_NO_THROW(g.cluster_node(2, 4));
}

TEST(Graph, classify_node_fail_no_finalize) {
  GDBSCAN::Graph g(5);
  ASSERT_THROW(g.cluster_node(0, 2), std::runtime_error);
}

TEST(Graph, classify_node_fail_oob) {
  GDBSCAN::Graph g(5);
  ASSERT_NO_THROW(g.finalize());
  ASSERT_NO_THROW(g.cluster_node(0, 2));
  ASSERT_THROW(g.cluster_node(-1, 2), std::runtime_error);
  ASSERT_THROW(g.cluster_node(6, 2), std::runtime_error);
}

TEST(TwoD, distance) {
  using namespace GDBSCAN::dimension;
  TwoD p(1, 2), q(3, 4);
  EXPECT_DOUBLE_EQ(p - q, std::sqrt(std::pow(1 - 3, 2) + std::pow(2 - 4, 2)));
  p = TwoD(-1, -4);
  q = TwoD(4, 1);
  EXPECT_DOUBLE_EQ(p - q, std::sqrt(std::pow(-1 - 4, 2) + std::pow(-4 - 1, 2)));
  p = TwoD(0, 0);
  EXPECT_DOUBLE_EQ(p - q, std::sqrt(std::pow(0 - 4, 2) + std::pow(0 - 1, 2)));
  p = TwoD(1, 2);
  q = TwoD(2.5, 3.4);
  EXPECT_DOUBLE_EQ(
      p - q, std::sqrt(std::pow(1.0f - 2.5f, 2) + std::pow(2.0f - 3.4f, 2)));
}

TEST(Dataset, setter_getter) {
  using namespace GDBSCAN::dimension;
  GDBSCAN::Dataset<TwoD> d(5);
  d[0] = TwoD(0.0f, 0.0f);
  d[1] = TwoD(1.0f, 1.0f);
  d[2] = TwoD(2.0f, 2.0f);
  d[3] = TwoD(3.0f, 3.0f);
  d[4] = TwoD(4.0f, 4.0f);
  EXPECT_THAT(d.view(), testing::ElementsAre(TwoD(0.0f, 0.0f),
                                             TwoD(1.0f, 1.0f),
                                             TwoD(2.0f, 2.0f),
                                             TwoD(3.0f, 3.0f),
                                             TwoD(4.0f, 4.0f)));
}

TEST(Runner, prepare_dataset) {
  auto ifs = std::make_unique<std::ifstream>("../test/test_input1.txt");
  int num_nodes;
  *ifs >> num_nodes;
  EXPECT_EQ(num_nodes, 6);

  using namespace GDBSCAN::dimension;
  GDBSCAN::Runner<TwoD> runner(num_nodes, 2, 3.0f);
  ASSERT_NO_THROW(runner.prepare_dataset(std::move(ifs)));
  EXPECT_THAT(runner.dataset_view(), testing::ElementsAre(TwoD(1.0f, 2.0f),
                                                          TwoD(2.0f, 2.0f),
                                                          TwoD(2.0f, 3.0f),
                                                          TwoD(8.0f, 7.0f),
                                                          TwoD(8.0f, 8.0f),
                                                          TwoD(25.0f, 80.0f)));
}

TEST(Runner, make_graph) {
  auto ifs = std::make_unique<std::ifstream>("../test/test_input1.txt");
  int num_nodes;
  *ifs >> num_nodes;
  EXPECT_EQ(num_nodes, 6);

  using namespace GDBSCAN::dimension;
  GDBSCAN::Runner<TwoD> runner(num_nodes, 2, 3.0f);
  ASSERT_NO_THROW(runner.prepare_dataset(std::move(ifs)));

  ASSERT_NO_THROW(runner.make_graph());
  auto graph = runner.graph_view();
  /*
   * Va:
   * 2 2 2 1 1 0 <- number of neighbours
   * 0 2 4 6 7 8 <- start pos in Ea
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
              testing::ElementsAre(2, 0, 2, 2, 2, 4, 1, 6, 1, 7, 0, 8));
  EXPECT_THAT(graph.Ea, testing::ElementsAre(1, 2, 0, 2, 0, 1, 4, 3));
}

TEST(Runner, classify_nodes) {
  auto ifs = std::make_unique<std::ifstream>("../test/test_input1.txt");
  int num_nodes;
  *ifs >> num_nodes;
  EXPECT_EQ(num_nodes, 6);

  using namespace GDBSCAN::dimension;
  GDBSCAN::Runner<TwoD> runner(num_nodes, 2, 3.0f);
  ASSERT_NO_THROW(runner.prepare_dataset(std::move(ifs)));
  ASSERT_NO_THROW(runner.make_graph());
  ASSERT_NO_THROW(runner.classify_nodes());
  auto graph = runner.graph_view();
  using namespace GDBSCAN::membership;
  EXPECT_THAT(graph.membership,
              testing::ElementsAre(Core, Core, Core, Noise, Noise, Noise));
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
