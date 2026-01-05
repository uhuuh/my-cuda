cmake_minimum_required(VERSION 3.10)
#set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_ARCHITECTURES "120")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")
project(bird VERSION 0.1 LANGUAGES CUDA CXX C)
find_package(CUDAToolkit REQUIRED)
add_subdirectory(sgemm)
add_subdirectory(memory)

