# Required packages and flags
## CUDA
set(CMAKE_CUDA_COMPILER "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4")
enable_language(CUDA)  # 启用CUDA支持
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message("CUDA FOUND, Version: " ${CUDA_VERSION_STRING})
endif ()
## OpenACC
find_package(OpenACC REQUIRED)
if (OpenACC_CXX_FOUND)
    message("OpenACC FOUND, Version: " ${OpenACC_CXX_VERSION})
endif ()

SET(CMAKE_CXX_COMPILER pgc++)

## CUDA
list(APPEND CUDA_NVCC_FLAGS "-fmad=false")
cuda_add_executable(cuda_PartA
        cuda_PartA.cu
        ../utils.cpp ../utils.hpp)
target_link_libraries(cuda_PartA cudart)

cuda_add_executable(cuda_PartB
        cuda_PartB.cu
        ../utils.cpp ../utils.hpp)
target_link_libraries(cuda_PartB cudart)

cuda_add_executable(cuda_PartC
        cuda_PartC.cu
        ../utils.cpp ../utils.hpp)
target_link_libraries(cuda_PartC cudart)

## OpenACC
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc -ta=tesla:nofma")
add_executable(openacc_PartA
        openacc_PartA.cpp
        ../utils.cpp ../utils.hpp)

add_executable(openacc_PartB
        openacc_PartB.cpp
        ../utils.cpp ../utils.hpp)

add_executable(openacc_PartC
        openacc_PartC.cpp
        ../utils.cpp ../utils.hpp)
