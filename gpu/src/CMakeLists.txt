add_library(GDBSCAN STATIC gdbscan.cu)
set_target_properties(GDBSCAN PROPERTIES LINKER_LANGUAGE CUDA)