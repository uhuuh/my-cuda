cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(cuda_only_project LANGUAGES CUDA)


file(GLOB_RECURSE CUDA_SOURCES
        CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
)

add_library(cuda_kernels STATIC ${CUDA_SOURCES})

#target_include_directories(cuda_kernels SYSTEM PRIVATE
#        #/mnt/c/Users/uh/code/work/1220-op/cutlass/include
#        /root/miniconda3/envs/py312/lib/python3.12/site-packages/torch/include/
#        /root/miniconda3/envs/py312/lib/python3.12/site-packages/torch/include/torch/csrc/api/include
#)
