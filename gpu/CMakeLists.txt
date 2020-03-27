# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# 1660Ti: sm_75, compute_75
# V100: sm_70, compute_70
# 1070: sm_61, compute_61
set(CUDA_ARCH arch=compute_61,code=sm_61)
set(CMAKE_CUDA_FLAGS "-gencode ${CUDA_ARCH} --ptxas-options=-O3 -O3")
set(CMAKE_CUDA_STANDARD 14)

include_directories(${CMAKE_SOURCE_DIR}/gpu/src)

add_subdirectory(${CMAKE_SOURCE_DIR}/gpu/src)
add_subdirectory(${CMAKE_SOURCE_DIR}/gpu/test)
add_subdirectory(${CMAKE_SOURCE_DIR}/gpu/main)