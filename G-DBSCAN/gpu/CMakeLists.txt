# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# 1660Ti: sm_75, compute_75
# 1070: sm_61, compute_61
set(CUDA_ARCH -gencode arch=compute_75,code=sm_75)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -O3 --std=c++14)
cuda_add_library(GDBSCAN STATIC gdbscan.cuh)
#set_target_properties(GDBSCAN PROPERTIES POSITION_INDEPENDENT_CODE ON)
#set_target_properties(GDBSCAN PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(gpu-main main.cu)
#set_target_properties(gpu-main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu-main PRIVATE GDBSCAN)