add_executable(cpu-main main.cpp)
target_link_libraries(cpu-main DBSCAN)
target_compile_definitions(cpu-main PRIVATE "${BIT_ADJ}" "${AVX}")
target_compile_definitions(DBSCAN PRIVATE "${BIT_ADJ}" "${AVX}")