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
add_executable(radixsort_sequential
        sequential.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(radixsort_sequential PRIVATE -O2)

## OpenMP
add_executable(radixsort_omp
        omp.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(radixsort_omp PRIVATE -O2 -fopenmp)
target_include_directories(radixsort_omp PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(radixsort_omp PRIVATE ${OpenMP_CXX_LIBRARIES})

## MPI
add_executable(radixsort_mpi
        mpi.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(radixsort_mpi PRIVATE -O2)
target_include_directories(radixsort_mpi PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(radixsort_mpi ${MPI_LIBRARIES})