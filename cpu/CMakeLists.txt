set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pthread -O3 -march=native -mavx")

if (DEFINED ENV{BIT_ADJ})
  message("*** using packed uint64_t adjacency matrix (vector<vector<std::uint64_t>> of size NxN/64)")
  set(BIT_ADJ BIT_ADJ)
endif ()

if (DEFINED ENV{AVX})
  message("*** enabling AVX")
  set(AVX AVX)
endif ()

# spdlog
include_directories(${CMAKE_SOURCE_DIR}/third_party/spdlog/include)

# DBSCAN
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/main)