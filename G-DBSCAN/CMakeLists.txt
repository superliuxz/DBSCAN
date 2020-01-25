cmake_minimum_required(VERSION 3.15)
project(GDBSCAN)

set(CMAKE_CXX_STANDARD 17)

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

# GDBSCAN lib
add_library(GDBSCAN SHARED include/Graph.h include/Dataset.h include/Dimension.h include/Runner.h include/Membership.h)
set_target_properties(GDBSCAN PROPERTIES LINKER_LANGUAGE CXX)

# test
enable_testing()
add_executable(unittest test/tests.cpp include/Dataset.h include/Dimension.h include/Runner.h include/Membership.h)
add_test(unittest test/test.cpp)
target_link_libraries(unittest gtest_main gmock_main GDBSCAN)

# example
add_executable(example example/example.cpp include/Dataset.h include/Dimension.h include/Runner.h include/Membership.h)
target_link_libraries(example GDBSCAN)