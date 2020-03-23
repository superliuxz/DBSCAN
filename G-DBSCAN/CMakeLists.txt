cmake_minimum_required(VERSION 3.10)
project(DBSCAN)

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

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# spdlog
include_directories(${CMAKE_SOURCE_DIR}/third_party/spdlog/include)

# cxxopts
include_directories(${CMAKE_SOURCE_DIR}/third_party/cxxopts/include)

# DBSCAN
include_directories("${CMAKE_SOURCE_DIR}/src")

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(main)
add_subdirectory(gpu)