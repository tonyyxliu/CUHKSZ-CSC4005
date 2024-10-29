## MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    message("MPI FOUND, Version: ${MPI_CXX_VERSION}")
endif ()

## Sequential
add_executable(psrs_mpi
        mpi.cpp
        ../utils.cpp
        ../utils.hpp)
target_compile_options(psrs_mpi PRIVATE -O2)
target_include_directories(psrs_mpi PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(psrs_mpi ${MPI_LIBRARIES})
