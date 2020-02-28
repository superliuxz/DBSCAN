cmake_minimum_required(VERSION 3.10)
project(GDBSCAN)

set(CMAKE_CXX_STANDARD 17)

message("***********************************************")
if (DEFINED ENV{BENCH})
  message("*** benchmark mode: -O0 -march=native *********")
  set(CMAKE_CXX_FLAGS "-Wall -Wextra -O0 -march=native")
else ()
  message("*** unittest: -O3 -march=native ***************")
  set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3 -march=native")
endif ()
message("***********************************************\n")

message("*****************************************************************************")

if (DEFINED ENV{TILING})
  message("*** apply tiling at nested loop of insert_edges ")
  set(TILING TILING)
endif ()

if (DEFINED ENV{BIT_ADJ_UNROLL})
  message("*** BIT_ADJ_UNROLL")
  set(BIT_ADJ_UNROLL BIT_ADJ_UNROLL)
endif ()

if (DEFINED ENV{UNROLL_INSE})
  message("*** unroll inner loop at insert_edges")
  set(UNROLL_INSE UNROLL_INSE)
endif ()

if (DEFINED ENV{TRIA_ENUM})
  message("*** triangle enumeration enabled!")
  set(TRIA_ENUM TRIA_ENUM)
else ()
  message("*** square enumeration")
endif ()

if (DEFINED ENV{FLAT_ADJ})
  message("*** using 1D flat adjacency matrix (vector<size_t> of size NxN)")
  set(FLAT_ADJ FLAT_ADJ)
elseif (DEFINED ENV{BOOL_ADJ}) # SMH http://www.gotw.ca/gotw/050.htm
  message("*** using 2D boolean adjacency matrix of size NxN")
  set(BOOL_ADJ BOOL_ADJ)
elseif (DEFINED ENV{BIT_ADJ})
  message("*** using packed uint64_t adjacency matrix (vector<vector<std::uint64_t>> of size NxN/64)")
  set(BIT_ADJ BIT_ADJ)
elseif (DEFINED ENV{BITSET_ADJ})
  message("*** using BITSET_ADJ adjacency matrix (vector<vector<bitset<64>>>)")
  set(BITSET_ADJ BITSET_ADJ)
else ()
  message("*** using default dynamic adjacency list (std::vector<std::vector<size_t>>)")
endif ()

if (DEFINED ENV{SQRE_RADIUS})
  message("*** square the radius to avoid square root computing (which uses DIV unit)")
  set(SQRE_RADIUS SQRE_RADIUS)
elseif (DEFINED ENV{HYPOTF})
  message("*** using std::hypotf to calculate radius")
  set(HYPOTF HYPOTF)
else ()
  message("*** using std::srqt with std::pow to calculate radius")
endif ()

if (DEFINED ENV{DFS})
  message("*** using DFS to cluster the graph")
  set(DFS DFS)
elseif (DEFINED ENV{UNROLL_BFS})
  message("*** unroll BFS to use ILP")
  set(UNROLL_BFS UNROLL_BFS)
endif ()

message("*****************************************************************************")
##
# Copied from
# https://github.com/google/googletest/tree/master/googletest#incorporating-into-an-existing-cmake-project
##
# Download and unpack googletest at configure time
configure_file(CMakeLists.GoogleTest.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
if (result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif ()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
if (result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif ()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
    ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
    EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif ()
##
# end of copy
##

# spdlog
include_directories(third_party/spdlog/include)

# cxxopts
include_directories(third_party/cxxopts/include)

# GDBSCAN lib
add_library(GDBSCAN SHARED include/Graph.h include/Point.h include/Solver.h include/Membership.h include/Helper.h)
set_target_properties(GDBSCAN PROPERTIES LINKER_LANGUAGE CXX)

# test
enable_testing()
add_executable(unittest test/tests.cpp include/Point.h include/Solver.h include/Membership.h)
add_test(unittest test/test.cpp)
target_link_libraries(unittest gtest_main gmock_main GDBSCAN)
target_compile_definitions(unittest PRIVATE TESTING)
target_compile_definitions(unittest PRIVATE "${TILING}" "${FLAT_ADJ}" "${TRIA_ENUM}" "${BIT_ADJ}" "${HYPOTF}" "${SQRE_RADIUS}" "${UNROLL_INSE}" "${BOOL_ADJ}" "${DFS}" "${UNROLL_BFS}" "${BITSET_ADJ}" "${BIT_ADJ_UNROLL}")

# example
add_executable(main main.cpp include/Point.h include/Solver.h include/Membership.h include/Helper.h)
target_link_libraries(main GDBSCAN)
target_compile_definitions(main PRIVATE "${TILING}" "${FLAT_ADJ}" "${TRIA_ENUM}" "${BIT_ADJ}" "${HYPOTF}" "${SQRE_RADIUS}" "${UNROLL_INSE}" "${BOOL_ADJ}" "${DFS}" "${UNROLL_BFS}" "${BITSET_ADJ}" "${BIT_ADJ_UNROLL}")
