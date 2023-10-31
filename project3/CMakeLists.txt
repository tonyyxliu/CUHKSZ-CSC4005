# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Project Name
project(
        CSC4005-Project-3
        VERSION 1.0
        DESCRIPTION
        "Distributed and Parallel Sorting Algorithms"
        LANGUAGES C CXX)

# C++11 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Require at least gcc/g++ 4.8.5) CMAKE_CXX_COMPILER_VERSION OR
if (CMAKE_C_COMPILER_VERSION VERSION_LESS 4.8.5 OR CMAKE_CXX_COMPILER_VERSION
        VERSION_LESS 4.8.5)
    message(
            FATAL_ERROR "gcc and g++ version => 4.8.5 is required for building project!")
endif ()

add_subdirectory(src)
