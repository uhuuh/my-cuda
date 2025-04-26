add_executable(a a.cu)
target_link_libraries(a PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(a PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


