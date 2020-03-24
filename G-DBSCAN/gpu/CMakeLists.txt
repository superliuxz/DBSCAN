# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# 1660Ti: sm_75, compute_75
# 1070: sm_61, compute_61
set(CUDA_ARCH arch=compute_75,code=sm_75)
set(CMAKE_CUDA_FLAGS "-gencode ${CUDA_ARCH} --ptxas-options=-v,-O3")
set(CMAKE_CUDA_STANDARD 14)
add_library(GDBSCAN gdbscan.cuh)
set_target_properties(GDBSCAN PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(gpu-main main.cu)
target_link_libraries(gpu-main GDBSCAN)