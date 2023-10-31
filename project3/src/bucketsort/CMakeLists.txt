## MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    message("MPI FOUND, Version: ${MPI_CXX_VERSION}")
endif ()

## Sequential
add_executable(bucketsort_sequential
        sequential.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(bucketsort_sequential PRIVATE -O2)

## MPI
add_executable(bucketsort_mpi
        mpi.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(bucketsort_mpi PRIVATE -O2)
target_include_directories(bucketsort_mpi PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(bucketsort_mpi ${MPI_LIBRARIES})