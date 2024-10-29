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
add_executable(naive_sparse
        naive.cpp
        sparse_matrix.cpp sparse_matrix.hpp)
target_compile_options(naive PRIVATE -O2)

## MPI
add_executable(parallelized_sparse
        parallelized.cpp
        sparse_matrix.cpp sparse_matrix.hpp)
target_compile_options(parallelized_sparse PRIVATE -O2 -fopenmp -mavx2)
target_include_directories(parallelized_sparse PRIVATE ${MPI_CXX_INCLUDE_DIRS} ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(parallelized_sparse ${MPI_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
