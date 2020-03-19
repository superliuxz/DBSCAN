add_executable(main main.cpp)
target_link_libraries(main GDBSCAN)
target_link_libraries(main)
target_compile_definitions(main PRIVATE "${BIT_ADJ}" "${AVX}")