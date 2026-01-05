add_executable(a a.cu)
target_link_libraries(a PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

add_executable(b ../reduuce/b.cu)
target_link_libraries(b PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

add_executable(c c.cu)
target_link_libraries(c PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
#if (CMAKE_BUILD_TYPE STREQUAL "Debug")
#    target_compile_options(a PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
#endif()


