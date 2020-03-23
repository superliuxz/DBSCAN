add_library(GDBSCAN SHARED solver.cpp graph.cpp)
set_target_properties(GDBSCAN PROPERTIES LINKER_LANGUAGE CXX)