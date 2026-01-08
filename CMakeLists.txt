cmake_minimum_required(VERSION 3.18)

project(cuda_only_project LANGUAGES CUDA)

include_directories(
        /mnt/c/Users/uh/code/work/1220-op/cutlass/include
        /root/miniconda3/envs/py312/lib/python3.12/site-packages/torch/include/
        /root/miniconda3/envs/py312/lib/python3.12/site-packages/torch/include/torch/csrc/api/include
)

file(GLOB_RECURSE CUDA_SOURCES
        CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
)

add_library(cuda_kernels STATIC ${CUDA_SOURCES})

set_target_properties(cuda_kernels PROPERTIES
        CUDA_ARCHITECTURES native
)
