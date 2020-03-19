add_library(GDBSCAN SHARED graph.h solver.h membership.h utils.h dataset.h solver.cpp graph.cpp)
set_target_properties(GDBSCAN PROPERTIES LINKER_LANGUAGE CXX)