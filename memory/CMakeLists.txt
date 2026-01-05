add_executable(memory memory.cu)
target_link_libraries(memory PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})



