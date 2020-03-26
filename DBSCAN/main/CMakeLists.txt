add_executable(main main.cpp)
target_link_libraries(main DBSCAN)
target_compile_definitions(main PRIVATE "${BIT_ADJ}" "${AVX}")