# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# 1660Ti: sm_75, compute_75
# 1070: sm_61, compute_61
set(CUDA_ARCH -gencode arch=compute_75,code=sm_75)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -O3 --std=c++14)
cuda_add_library(GDBSCAN gdbscan.cuh)
# force nvcc to compile *.cu, *.cuh files
set_source_files_properties(main.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
cuda_add_executable(gpu-main main.cpp)
target_link_libraries(gpu-main GDBSCAN)