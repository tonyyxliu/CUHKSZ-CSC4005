# Required packages and flags
## MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    message("MPI FOUND, Version: ${MPI_CXX_VERSION}")
endif ()
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

## SIMD Vectorization (AVX2)
add_executable(simd
        simd.cpp
        matrix.cpp matrix.hpp)
target_compile_options(simd PRIVATE -O2 -mavx2)

## OpenMP
add_executable(openmp
        openmp.cpp
        matrix.cpp matrix.hpp)
target_compile_options(openmp PRIVATE -O2 -fopenmp -mavx2)
target_include_directories(openmp PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(openmp PRIVATE ${OpenMP_CXX_LIBRARIES})

## MPI
add_executable(mpi
        mpi.cpp
        matrix.cpp matrix.hpp)
target_compile_options(mpi PRIVATE -O2 -fopenmp -mavx2)
target_include_directories(mpi PRIVATE ${MPI_CXX_INCLUDE_DIRS} ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(mpi ${MPI_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
