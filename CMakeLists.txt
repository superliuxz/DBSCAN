cmake_minimum_required(VERSION 3.10)
project(DBSCAN LANGUAGES CXX CUDA)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# cxxopts
include_directories(${CMAKE_SOURCE_DIR}/third_party/cxxopts/include)

# shared across CPU and GPU implementations
include_directories(${CMAKE_SOURCE_DIR}/shared)

add_subdirectory(${CMAKE_SOURCE_DIR}/gtest)
add_subdirectory(${CMAKE_SOURCE_DIR}/cpu)
add_subdirectory(${CMAKE_SOURCE_DIR}/gpu)