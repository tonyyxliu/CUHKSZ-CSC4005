# Required packages and flags
## CUDA
set(CMAKE_CUDA_COMPILER "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4")
enable_language(CUDA)  # 启用CUDA支持
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message("CUDA FOUND, Version: " ${CUDA_VERSION_STRING})
endif ()

## CUDA
cuda_add_executable(cuda_radixsort cuda_radixsort.cu ../utils.cpp ../utils.hpp)
target_compile_options(cuda_radixsort PRIVATE -O3)
target_link_libraries(cuda_radixsort)
