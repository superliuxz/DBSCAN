add_executable(cpu-main main.cpp)
target_link_libraries(cpu-main DBSCAN)
target_compile_definitions(DBSCAN PUBLIC DBSCAN_TESTING=false "${BIT_ADJ}" "${AVX}")