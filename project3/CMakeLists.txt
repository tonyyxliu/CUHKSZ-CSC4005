# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Project Name
project(
        CSC4005-Project-3
        VERSION 1.0
        DESCRIPTION
        "Parallel Sorting and Searching Algorithms"
        LANGUAGES C CXX)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Require at least gcc/g++ 11.4.0) CMAKE_CXX_COMPILER_VERSION OR
if (CMAKE_C_COMPILER_VERSION VERSION_LESS 11.4.0 OR CMAKE_CXX_COMPILER_VERSION
        VERSION_LESS 11.4.0)
    message(
            FATAL_ERROR "gcc and g++ version => 11.4.0 is required for building project!")
endif ()

add_subdirectory(src)
