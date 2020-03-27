add_library(DBSCAN SHARED solver.cpp graph.cpp)
set_target_properties(DBSCAN PROPERTIES LINKER_LANGUAGE CXX)