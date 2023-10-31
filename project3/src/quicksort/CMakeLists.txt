find_package(MPI REQUIRED)
if (MPI_FOUND)
    message("MPI FOUND, Version: ${MPI_CXX_VERSION}")
endif ()

## OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    message("OPENMP FOUND, Version: " ${OpenMP_CXX_VERSION})
endif ()

## Sequential
add_executable(quicksort_sequential
        sequential.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(quicksort_sequential PRIVATE -O2)

## Parallel
add_executable(quicksort_parallel
        parallel.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(quicksort_parallel PRIVATE -O2 -fopenmp)
target_include_directories(quicksort_parallel PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(quicksort_parallel PRIVATE ${OpenMP_CXX_LIBRARIES})

## MPI
add_executable(quicksort_mpi
        mpi.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(quicksort_mpi PRIVATE -O2)
target_include_directories(quicksort_mpi PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(quicksort_mpi ${MPI_LIBRARIES})