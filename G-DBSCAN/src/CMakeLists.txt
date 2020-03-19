include_directories("${CMAKE_SOURCE_DIR}/src")
add_library(GDBSCAN SHARED solver.cpp graph.cpp)
set_target_properties(GDBSCAN PROPERTIES LINKER_LANGUAGE CXX)