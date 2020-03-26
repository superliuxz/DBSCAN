# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# 1660Ti: sm_75, compute_75
# V100: sm_70, compute_70
# 1070: sm_61, compute_61
include_directories(${CMAKE_SOURCE_DIR}/gpu)
set(CUDA_ARCH arch=compute_61,code=sm_61)
set(CMAKE_CUDA_FLAGS "-gencode ${CUDA_ARCH} --ptxas-options=-v,-O3 -O3")
set(CMAKE_CUDA_STANDARD 14)
add_library(GDBSCAN gdbscan.cuh)
set_target_properties(GDBSCAN PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(gpu-main main.cu)
target_link_libraries(gpu-main GDBSCAN)