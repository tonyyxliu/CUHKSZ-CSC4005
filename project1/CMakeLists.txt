# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Project Name
project(
        CSC4005/MDS6108-Project-1
        VERSION 1.0
        DESCRIPTION
        "Programming Project 1 for CSC4005/MDS6108: Parallel Programming"
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

add_subdirectory(src)
