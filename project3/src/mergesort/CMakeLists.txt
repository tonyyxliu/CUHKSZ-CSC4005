## OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    message("OPENMP FOUND, Version: " ${OpenMP_CXX_VERSION})
endif ()

## Sequential
add_executable(mergesort_sequential
        sequential.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(mergesort_sequential PRIVATE -O2)

## Parallel
add_executable(mergesort_parallel
        parallel.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(mergesort_parallel PRIVATE -O2 -fopenmp)
target_include_directories(mergesort_parallel PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(mergesort_parallel PRIVATE ${OpenMP_CXX_LIBRARIES})