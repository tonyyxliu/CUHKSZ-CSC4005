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

## OpenMP Tasking
add_executable(mergesort
        mergesort.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(mergesort PRIVATE -O2 -fopenmp)
target_include_directories(mergesort PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(mergesort PRIVATE ${OpenMP_CXX_LIBRARIES})
