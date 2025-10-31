## OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    message("OPENMP FOUND, Version: " ${OpenMP_CXX_VERSION})
endif ()

## Task #4: Sequential Array Binary Search
add_executable(searching_array_sequential
        sequential_array.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(searching_array_sequential PRIVATE -O2)

## Task #4: Parallel Array Binary Search
add_executable(searching_array_parallel
        parallel_array.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(searching_array_parallel PRIVATE -O2 -fopenmp)
target_include_directories(searching_array_parallel PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(searching_array_parallel PRIVATE ${OpenMP_CXX_LIBRARIES})
