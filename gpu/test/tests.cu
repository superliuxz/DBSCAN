//
// Created by will on 2020-03-26.
//

#include <gmock/gmock.h>  // ASSERT_THAT, testing::ElementsAre
#include <gtest/gtest.h>

#include <fstream>

#include "gdbscan.cuh"

namespace DBSCAN_TestVariables {
std::string abs_loc;
}

class GDBSCAN_TestEnvironment : public testing::Environment {
 public:
  explicit GDBSCAN_TestEnvironment(const std::string& command_line_arg) {
    DBSCAN_TestVariables::abs_loc = command_line_arg;
  }
};

TEST(Solver, make_graph_small_graph) {
  using namespace GDBSCAN;
  Solver solver(DBSCAN_TestVariables::abs_loc + "/test_input1.txt", 2, 3.0f);
  ASSERT_NO_THROW(solver.calc_num_neighbours());
  ASSERT_NO_THROW(solver.calc_start_pos());
  ASSERT_NO_THROW(solver.append_neighbours());
  ASSERT_NO_THROW(solver.identify_cores());
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
  EXPECT_THAT(solver.num_neighbours, testing::ElementsAre(2, 2, 2, 1, 1, 0));
  EXPECT_THAT(solver.start_pos, testing::ElementsAre(0, 2, 4, 6, 7, 8));
  EXPECT_THAT(solver.neighbours, testing::ElementsAre(1, 2, 0, 2, 0, 1, 4, 3));
  EXPECT_THAT(solver.memberships,
              testing::ElementsAre(
                  DBSCAN::membership::Core, DBSCAN::membership::Core,
                  DBSCAN::membership::Core, DBSCAN::membership::Noise,
                  DBSCAN::membership::Noise, DBSCAN::membership::Noise));
}

TEST(Solver, test_input1) {
  using namespace GDBSCAN;
  Solver solver(DBSCAN_TestVariables::abs_loc + "/test_input1.txt", 2, 3.0f);
  ASSERT_NO_THROW(solver.calc_num_neighbours());
  ASSERT_NO_THROW(solver.calc_start_pos());
  ASSERT_NO_THROW(solver.append_neighbours());
  ASSERT_NO_THROW(solver.identify_cores());
  ASSERT_NO_THROW(solver.identify_clusters());
  // vertices 0 1 and 2 are core vertices with cluster id = 0; vertices 3 4 and
  // 5 are noise vertices hence cluster id = -1.
  EXPECT_THAT(solver.cluster_ids, testing::ElementsAre(0, 0, 0, -1, -1, -1));
}

TEST(Solver, test_input2) {
  using namespace GDBSCAN;
  Solver solver(DBSCAN_TestVariables::abs_loc + "/test_input2.txt", 2, 3.0f);
  ASSERT_NO_THROW(solver.calc_num_neighbours());
  ASSERT_NO_THROW(solver.calc_start_pos());
  ASSERT_NO_THROW(solver.append_neighbours());
  ASSERT_NO_THROW(solver.identify_cores());
  ASSERT_NO_THROW(solver.identify_clusters());
  EXPECT_THAT(solver.memberships,
              testing::ElementsAre(DBSCAN::membership::Core,    // 0
                                   DBSCAN::membership::Core,    // 1
                                   DBSCAN::membership::Core,    // 2
                                   DBSCAN::membership::Border,  // 3
                                   DBSCAN::membership::Border,  // 4
                                   DBSCAN::membership::Core,    // 5
                                   DBSCAN::membership::Core,    // 6
                                   DBSCAN::membership::Core,    // 7
                                   DBSCAN::membership::Core,    // 8
                                   DBSCAN::membership::Noise    // 9
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
  using namespace GDBSCAN;
  Solver solver(DBSCAN_TestVariables::abs_loc + "/test_input3.txt", 3, 3.0f);
  ASSERT_NO_THROW(solver.calc_num_neighbours());
  ASSERT_NO_THROW(solver.calc_start_pos());
  ASSERT_NO_THROW(solver.append_neighbours());
  ASSERT_NO_THROW(solver.identify_cores());
  ASSERT_NO_THROW(solver.identify_clusters());
  EXPECT_THAT(solver.memberships,
              testing::ElementsAre(DBSCAN::membership::Core,    // 0
                                   DBSCAN::membership::Core,    // 1
                                   DBSCAN::membership::Core,    // 2
                                   DBSCAN::membership::Core,    // 3
                                   DBSCAN::membership::Border,  // 4
                                   DBSCAN::membership::Noise,   // 5
                                   DBSCAN::membership::Border,  // 6
                                   DBSCAN::membership::Core,    // 7
                                   DBSCAN::membership::Core,    // 8
                                   DBSCAN::membership::Core,    // 9
                                   DBSCAN::membership::Core     // 10
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
  using namespace GDBSCAN;
  Solver solver(DBSCAN_TestVariables::abs_loc + "/test_input_20k.txt", 30,
                0.15f);
  ASSERT_NO_THROW(solver.calc_num_neighbours());
  ASSERT_NO_THROW(solver.calc_start_pos());
  ASSERT_NO_THROW(solver.append_neighbours());
  ASSERT_NO_THROW(solver.identify_cores());
  ASSERT_NO_THROW(solver.identify_clusters());
  std::vector<int> expected_labels;
  std::ifstream ifs(DBSCAN_TestVariables::abs_loc +
                    "/test_input_20k_labels.txt");
  int label;
  while (ifs >> label) expected_labels.push_back(label);
  EXPECT_THAT(solver.cluster_ids, testing::ElementsAreArray(expected_labels));
}

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  testing::AddGlobalTestEnvironment(new GDBSCAN_TestEnvironment(argv[1]));
  return RUN_ALL_TESTS();
}
