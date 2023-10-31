## MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    message("MPI FOUND, Version: ${MPI_CXX_VERSION}")
endif ()

## Sequential
add_executable(odd-even-sort_sequential
        sequential.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(odd-even-sort_sequential PRIVATE -O2)

## MPI
add_executable(odd-even-sort_mpi
        mpi.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(odd-even-sort_mpi PRIVATE -O2)
target_include_directories(odd-even-sort_mpi PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(odd-even-sort_mpi ${MPI_LIBRARIES})