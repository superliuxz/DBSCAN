add_library(DBSCAN STATIC solver.cpp graph.cpp grid.cpp)
set_target_properties(DBSCAN PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions(DBSCAN PRIVATE "${BIT_ADJ}" "${AVX}")