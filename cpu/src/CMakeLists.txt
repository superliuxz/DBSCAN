add_library(DBSCAN STATIC solver.cpp graph.cpp)
set_target_properties(DBSCAN PROPERTIES LINKER_LANGUAGE CXX)