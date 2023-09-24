# Required packages and flags
## MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    message("MPI FOUND, Version: ${MPI_CXX_VERSION}")
endif ()
## Pthread
find_package(Threads REQUIRED)
if (Threads_FOUND)
    message("Pthread FOUND")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
endif ()
## OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    message("OPENMP FOUND, Version: " ${OpenMP_CXX_VERSION})
endif ()

# Add Executables
## Sequential
add_executable(sequential_PartA
        sequential_PartA.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(sequential_PartA PRIVATE -O2)

add_executable(sequential_PartB
        sequential_PartB.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(sequential_PartB PRIVATE -O2)

## SIMD Vectorization (AVX2)
add_executable(simd_PartA
        simd_PartA.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(simd_PartA PRIVATE -O2 -mavx2)

## MPI
add_executable(mpi_PartA
        mpi_PartA.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(mpi_PartA PRIVATE -O2)
target_include_directories(mpi_PartA PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(mpi_PartA ${MPI_LIBRARIES})

## Pthread
add_executable(pthread_PartA
        pthread_PartA.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(pthread_PartA PRIVATE -O2)
target_link_libraries(pthread_PartA PRIVATE pthread)

## OpenMP
add_executable(openmp_PartA
        openmp_PartA.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(openmp_PartA PRIVATE -O2 -fopenmp)
target_include_directories(openmp_PartA PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(openmp_PartA PRIVATE ${OpenMP_CXX_LIBRARIES})
