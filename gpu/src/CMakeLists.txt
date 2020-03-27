add_library(GDBSCAN SHARED gdbscan.cu)
set_target_properties(GDBSCAN PROPERTIES LINKER_LANGUAGE CUDA)