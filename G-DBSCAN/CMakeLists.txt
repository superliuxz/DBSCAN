cmake_minimum_required(VERSION 3.10)
project(GDBSCAN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3")

if (DEFINED ENV{TILING})
  message("*** tiling enabled!")
  set(TILING TILING)
endif ()

if (DEFINED ENV{SQRE_ENUM})
  message("*** square enumeration enabled!")
  set(SQRE_ENUM SQRE_ENUM)
endif ()

if (DEFINED ENV{OPTM_1})
  message("Optimisation in Graph.h - finalize() enabled!")
  set(OPTM_1 OPTM_1)
endif ()

if (DEFINED ENV{FLAT_ADJ})
  message("*** using flat adjacency list to construct Ea and Va!")
  set(FLAT_ADJ FLAT_ADJ)
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
add_library(GDBSCAN SHARED include/Graph.h include/Point.h include/Solver.h include/Membership.h include/Helper.h)
set_target_properties(GDBSCAN PROPERTIES LINKER_LANGUAGE CXX)

# test
enable_testing()
add_executable(unittest test/tests.cpp include/Point.h include/Solver.h include/Membership.h)
add_test(unittest test/test.cpp)
target_link_libraries(unittest gtest_main gmock_main GDBSCAN)
target_compile_definitions(unittest PRIVATE TESTING)
target_compile_definitions(unittest PRIVATE "${TILING}" "${OPTM_1}" "${FLAT_ADJ}" "${SQRE_ENUM}")

# example
add_executable(main main.cpp include/Point.h include/Solver.h include/Membership.h include/Helper.h)
target_link_libraries(main GDBSCAN)
target_compile_definitions(main PRIVATE "${TILING}" "${OPTM_1}" "${FLAT_ADJ}" "${SQRE_ENUM}")