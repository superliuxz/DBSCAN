//
// Created by William Liu on 2020-01-23.
//

#include <gtest/gtest.h>
#include <gmock/gmock.h> // ASSERT_THAT, testing::ElementsAre
#include "../include/Graph.h"

TEST(ctor, success) {
  GDBSCAN::Graph<uint32_t> g(5);
  EXPECT_EQ(g.Va.size(), 10);
  EXPECT_EQ(g.cluster_ids.size(), 5);
  EXPECT_TRUE(g.Ea.empty());
}

TEST(insert_edge, success) {
  GDBSCAN::Graph<uint32_t> g(5);
  ASSERT_NO_THROW(g.insert_edge(2, 1));
  ASSERT_NO_THROW(g.insert_edge(2, 4));
  ASSERT_NO_THROW(g.insert_edge(2, 0));
  ASSERT_NO_THROW(g.insert_edge(0, 3));
}

TEST(insert_edge, failed_oob) {
  GDBSCAN::Graph<uint32_t> g(5);
  ASSERT_NO_THROW(g.insert_edge(2, 1));
  ASSERT_THROW(g.insert_edge(0, 5), std::runtime_error);
  ASSERT_THROW(g.insert_edge(-1, 2), std::runtime_error);
  ASSERT_THROW(g.insert_edge(-2, 9), std::runtime_error);
}

TEST(finalize, success) {
  GDBSCAN::Graph<uint32_t> g(5);
  ASSERT_NO_THROW(g.insert_edge(2, 1));
  ASSERT_NO_THROW(g.insert_edge(2, 4));
  ASSERT_NO_THROW(g.insert_edge(2, 0));
  ASSERT_NO_THROW(g.insert_edge(0, 3));
  ASSERT_NO_THROW(g.finalize());

  ASSERT_THAT(g.Va, testing::ElementsAre(2, 0, 1, 2, 3, 3, 1, 6, 1, 7));
  ASSERT_THAT(g.Ea, testing::ElementsAre(2, 3, 2, 1, 4, 0, 0, 2));
}

TEST(finalize, fail_second_finalize) {
  GDBSCAN::Graph<uint32_t> g(5);
  ASSERT_NO_THROW(g.insert_edge(2, 1));
  ASSERT_NO_THROW(g.insert_edge(2, 4));
  ASSERT_NO_THROW(g.insert_edge(2, 0));
  ASSERT_NO_THROW(g.insert_edge(0, 3));
  ASSERT_NO_THROW(g.finalize());

  ASSERT_THROW(g.finalize(), std::runtime_error);
}

TEST(finalize, success_disconnected_graph) {
  GDBSCAN::Graph<uint32_t> g(5);
  ASSERT_NO_THROW(g.insert_edge(2, 1));
  ASSERT_NO_THROW(g.insert_edge(2, 4));
  ASSERT_NO_THROW(g.insert_edge(2, 0));
  ASSERT_NO_THROW(g.finalize());

  ASSERT_THAT(g.Va, testing::ElementsAre(1, 0, 1, 1, 3, 2, 0, 5, 1, 5));
  ASSERT_THAT(g.Ea, testing::ElementsAre(2, 2, 1, 4, 0, 2));
}

TEST(finalize, success_no_edges) {
  GDBSCAN::Graph<uint32_t> g(5);
  ASSERT_NO_THROW(g.finalize());
  ASSERT_THAT(g.Va, testing::ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  ASSERT_TRUE(g.Ea.empty());
}

TEST(classify_node, success) {
  GDBSCAN::Graph<uint32_t> g(5);
  ASSERT_NO_THROW(g.finalize());
  ASSERT_NO_THROW(g.classify_node(0, 2));
  ASSERT_NO_THROW(g.classify_node(1, 2));
  ASSERT_NO_THROW(g.classify_node(2, 4));
}

TEST(classify_node, fail_no_finalize) {
  GDBSCAN::Graph<uint32_t> g(5);
  ASSERT_THROW(g.classify_node(0, 2), std::runtime_error);
}

TEST(classify_node, fail_oob) {
  GDBSCAN::Graph<uint32_t> g(5);
  ASSERT_NO_THROW(g.finalize());
  ASSERT_NO_THROW(g.classify_node(0, 2));
  ASSERT_THROW(g.classify_node(-1, 2), std::runtime_error);
  ASSERT_THROW(g.classify_node(6, 2), std::runtime_error);
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
