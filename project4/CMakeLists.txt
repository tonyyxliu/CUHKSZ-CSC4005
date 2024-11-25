# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Project Name
project(
        CSC4005/CSC6115-Project-4
        VERSION 1.0
        DESCRIPTION
        "Programming Project 4 for CSC4005/CSC6115: Parallel Programming"
        LANGUAGES CXX)

# C++11 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message("CMAKE_C_COMPILER_VERSION: ${CMAKE_C_COMPILER_VERSION}")
message("CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")

# Require at least gcc/g++ 4.8.5 CMAKE_CXX_COMPILER_VERSION OR
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.1)
    message(FATAL_ERROR "gcc and g++ version => 7.3.1 is required for building project!")
endif ()

# sequential
add_executable(sequential
        src/mlp_main.cpp 
        src/mlp_sequential.cpp 
        src/ops_sequential.cpp 
        src/utils.cpp)
target_include_directories(sequential PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(sequential PRIVATE -O2)

# OpenACC
find_package(OpenACC REQUIRED)
if (OpenACC_CXX_FOUND)
    message("OpenACC FOUND, Version: " ${OpenACC_CXX_VERSION})
endif ()

set(OPENACC_KERNEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mlp_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mlp_openacc_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ops_openacc_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
)

set(OPENACC_FUSION_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mlp_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mlp_openacc_fusion.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ops_openacc_fusion.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
)

add_custom_target(openacc_kernel ALL
    COMMAND pgc++ -acc -I${CMAKE_CURRENT_SOURCE_DIR}/src
            ${OPENACC_KERNEL_SOURCES}
            -o openacc_kernel
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Building openacc_kernel with pgc++"
)

add_custom_target(openacc_fusion ALL
    COMMAND pgc++ -acc -I${CMAKE_CURRENT_SOURCE_DIR}/src
            ${OPENACC_FUSION_SOURCES}
            -o openacc_fusion
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Building openacc_fusion with pgc++"
)
