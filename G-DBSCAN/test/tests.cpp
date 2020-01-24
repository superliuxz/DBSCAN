//
// Created by William Liu on 2020-01-23.
//

#include <gmock/gmock.h> // ASSERT_THAT, testing::ElementsAre
#include <gtest/gtest.h>

#include "../include/Dataset.h"
#include "../include/Dimension.h"
#include "../include/Graph.h"

TEST(Graph, ctor_success) {
  GDBSCAN::Graph<uint8_t> g(5);
  EXPECT_EQ(g.Va.size(), 10);
  EXPECT_EQ(g.cluster_ids.size(), 5);
  EXPECT_TRUE(g.Ea.empty());
}

TEST(Graph, ctor_wrong_datatype) {
  ASSERT_THROW(GDBSCAN::Graph<int> g(5), std::runtime_error);
}

TEST(Graph, insert_edge_success) {
  GDBSCAN::Graph<uint8_t> g(5);
  ASSERT_NO_THROW(g.insert_edge(2, 1));
  ASSERT_NO_THROW(g.insert_edge(2, 4));
  ASSERT_NO_THROW(g.insert_edge(2, 0));
  ASSERT_NO_THROW(g.insert_edge(0, 3));
}

TEST(Graph, insert_edge_failed_oob) {
  GDBSCAN::Graph<uint8_t> g(5);
  ASSERT_NO_THROW(g.insert_edge(2, 1));
  ASSERT_THROW(g.insert_edge(0, 5), std::runtime_error);
  ASSERT_THROW(g.insert_edge(-1, 2), std::runtime_error);
  ASSERT_THROW(g.insert_edge(-2, 9), std::runtime_error);
}

TEST(Graph, finalize_success) {
  GDBSCAN::Graph<uint8_t> g(5);
  ASSERT_NO_THROW(g.insert_edge(2, 1));
  ASSERT_NO_THROW(g.insert_edge(2, 4));
  ASSERT_NO_THROW(g.insert_edge(2, 0));
  ASSERT_NO_THROW(g.insert_edge(0, 3));
  ASSERT_NO_THROW(g.finalize());

  ASSERT_THAT(g.Va, testing::ElementsAre(2, 0, 1, 2, 3, 3, 1, 6, 1, 7));
  ASSERT_THAT(g.Ea, testing::ElementsAre(2, 3, 2, 1, 4, 0, 0, 2));
}

TEST(Graph, finalize_fail_second_finalize) {
  GDBSCAN::Graph<uint8_t> g(5);
  ASSERT_NO_THROW(g.insert_edge(2, 1));
  ASSERT_NO_THROW(g.insert_edge(2, 4));
  ASSERT_NO_THROW(g.insert_edge(2, 0));
  ASSERT_NO_THROW(g.insert_edge(0, 3));
  ASSERT_NO_THROW(g.finalize());

  ASSERT_THROW(g.finalize(), std::runtime_error);
}

TEST(Graph, finalize_success_disconnected_graph) {
  GDBSCAN::Graph<uint8_t> g(5);
  ASSERT_NO_THROW(g.insert_edge(2, 1));
  ASSERT_NO_THROW(g.insert_edge(2, 4));
  ASSERT_NO_THROW(g.insert_edge(2, 0));
  ASSERT_NO_THROW(g.finalize());

  ASSERT_THAT(g.Va, testing::ElementsAre(1, 0, 1, 1, 3, 2, 0, 5, 1, 5));
  ASSERT_THAT(g.Ea, testing::ElementsAre(2, 2, 1, 4, 0, 2));
}

TEST(Graph, finalize_success_no_edges) {
  GDBSCAN::Graph<uint8_t> g(5);
  ASSERT_NO_THROW(g.finalize());
  ASSERT_THAT(g.Va, testing::ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  ASSERT_TRUE(g.Ea.empty());
}

TEST(Graph, classify_node_success) {
  GDBSCAN::Graph<uint8_t> g(5);
  ASSERT_NO_THROW(g.finalize());
  ASSERT_NO_THROW(g.classify_node(0, 2));
  ASSERT_NO_THROW(g.classify_node(1, 2));
  ASSERT_NO_THROW(g.classify_node(2, 4));
}

TEST(Graph, classify_node_fail_no_finalize) {
  GDBSCAN::Graph<uint8_t> g(5);
  ASSERT_THROW(g.classify_node(0, 2), std::runtime_error);
}

TEST(Graph, classify_node_fail_oob) {
  GDBSCAN::Graph<uint8_t> g(5);
  ASSERT_NO_THROW(g.finalize());
  ASSERT_NO_THROW(g.classify_node(0, 2));
  ASSERT_THROW(g.classify_node(-1, 2), std::runtime_error);
  ASSERT_THROW(g.classify_node(6, 2), std::runtime_error);
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

TEST(Dataset, exclude_in_dataset) {
  using namespace GDBSCAN::dimension;
  GDBSCAN::Dataset<TwoD> d(5);
  ASSERT_NO_THROW(d.exclude(0));
  ASSERT_NO_THROW(d.exclude(1));
  ASSERT_NO_THROW(d.exclude(4));
  EXPECT_EQ(d.in_dataset(0), false);
  EXPECT_EQ(d.in_dataset(1), false);
  EXPECT_EQ(d.in_dataset(2), true);
  EXPECT_EQ(d.in_dataset(3), true);
  EXPECT_EQ(d.in_dataset(4), false);
}

TEST(Dataset, exclude_in_dataset_oob) {
  using namespace GDBSCAN::dimension;
  GDBSCAN::Dataset<TwoD> d(5);
  ASSERT_THROW(d.exclude(-1), std::runtime_error);
  ASSERT_THROW(d.in_dataset(-1), std::runtime_error);
  ASSERT_THROW(d.exclude(5), std::runtime_error);
  ASSERT_THROW(d.in_dataset(5), std::runtime_error);
}

TEST(Dataset, setter_getter) {
  using namespace GDBSCAN::dimension;
  GDBSCAN::Dataset<TwoD> d(5);
  d[0] = TwoD(0.0f, 0.0f);
  d[1] = TwoD(1.0f, 1.0f);
  d[2] = TwoD(2.0f, 2.0f);
  d[3] = TwoD(3.0f, 3.0f);
  d[4] = TwoD(4.0f, 4.0f);
  EXPECT_EQ(d.in_dataset(0), true);
  EXPECT_EQ(d.in_dataset(1), true);
  EXPECT_EQ(d.in_dataset(2), true);
  EXPECT_EQ(d.in_dataset(3), true);
  EXPECT_EQ(d.in_dataset(4), true);
  EXPECT_EQ(d[0], TwoD(0.0f, 0.0f));
  EXPECT_EQ(d[1], TwoD(1.0f, 1.0f));
  EXPECT_EQ(d[2], TwoD(2.0f, 2.0f));
  EXPECT_EQ(d[3], TwoD(3.0f, 3.0f));
  EXPECT_EQ(d[4], TwoD(4.0f, 4.0f));
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
