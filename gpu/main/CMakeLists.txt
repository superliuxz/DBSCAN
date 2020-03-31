add_executable(gpu-main main.cu)
target_link_libraries(gpu-main GDBSCAN)
target_compile_definitions(GDBSCAN PUBLIC GDBSCAN_TESTING=0)