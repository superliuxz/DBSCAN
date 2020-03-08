cmake_minimum_required(VERSION 3.10)
project(GDBSCAN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pthread -O3 -march=native -mavx")

if (DEFINED ENV{BIT_ADJ})
  message("*** using packed uint64_t adjacency matrix (vector<vector<std::uint64_t>> of size NxN/64)")
  set(BIT_ADJ BIT_ADJ)
else ()
  message("*** using default dynamic adjacency list (std::vector<std::vector<size_t>>)")
endif ()

if (DEFINED ENV{AVX})
  message("*** enabling AVX")
  set(AVX AVX)
endif ()

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
add_library(GDBSCAN SHARED include/graph.h include/solver.h include/membership.h include/utils.h include/dataset.h)
set_target_properties(GDBSCAN PROPERTIES LINKER_LANGUAGE CXX)

# test
enable_testing()
add_executable(unittest test/tests.cpp include/solver.h include/membership.h include/dataset.h)
add_test(unittest test/test.cpp)
target_link_libraries(unittest gtest_main gmock_main GDBSCAN)
target_compile_definitions(unittest PRIVATE TESTING)
target_compile_definitions(unittest PRIVATE "${BIT_ADJ}" "${AVX}")

# example
add_executable(main main.cpp include/solver.h include/membership.h include/utils.h include/dataset.h)
target_link_libraries(main GDBSCAN)
target_compile_definitions(main PRIVATE "${BIT_ADJ}" "${AVX}")
