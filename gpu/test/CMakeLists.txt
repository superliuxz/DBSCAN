enable_testing()
add_executable(gpu-test tests.cu)
add_test(gpu-test tests.cu)
target_link_libraries(gpu-test gtest_main gmock_main GDBSCAN)