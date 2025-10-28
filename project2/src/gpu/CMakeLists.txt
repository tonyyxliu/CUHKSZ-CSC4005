project(myPJ)
cmake_minimum_required(VERSION 3.22)
set(CMAKE_CUDA_COMPILER "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/compilers/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/12.2")

enable_language(CUDA)
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message("CUDA FOUND, Version: " ${CUDA_VERSION_STRING})
endif()

set(SOURCES
    cuda.cu
    kernel_launch.cu 
    matrix_lowPre.cpp
    kernels/shared_mem_tiling.cu
    kernels/1D_tiling.cu
    )
cuda_add_executable(matMul ${SOURCES})
target_include_directories(matMul PRIVATE ${CMAKE_SOURCE_DIR}/kernels ${CMAKE_SOURCE_DIR})
