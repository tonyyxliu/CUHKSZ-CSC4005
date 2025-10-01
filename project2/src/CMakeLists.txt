# Required packages and flags
## OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    message("OPENMP FOUND, Version: " ${OpenMP_CXX_VERSION})
endif ()

# Add Executables
## Naive
add_executable(naive
        naive.cpp
        matrix.cpp matrix.hpp)
target_compile_options(naive PRIVATE -O2)

## Memory Locality
add_executable(locality
        locality.cpp
        matrix.cpp matrix.hpp)
target_compile_options(locality PRIVATE -O2)

## Auto Vectorization
add_executable(autovec
        data_parallel.cpp
        matrix.cpp matrix.hpp)
target_compile_options(autovec PRIVATE -O2 -ftree-vectorize -fopt-info-vec -mavx2)

## OpenMP
add_executable(openmp
        openmp.cpp
        matrix.cpp matrix.hpp)
target_compile_options(openmp PRIVATE -O2 -ftree-vectorize -fopt-info-vec -fopenmp -mavx2)
target_include_directories(openmp PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(openmp PRIVATE ${OpenMP_CXX_LIBRARIES})
