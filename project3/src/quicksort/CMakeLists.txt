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
add_executable(quicksort
        quicksort.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(quicksort PRIVATE -O2 -fopenmp)
target_include_directories(quicksort PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(quicksort PRIVATE ${OpenMP_CXX_LIBRARIES})