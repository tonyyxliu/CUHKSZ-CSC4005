## std::sort Sequential
add_executable(std_sort
        std_sort.cpp
        utils.cpp utils.hpp)
target_compile_options(std_sort PRIVATE -O2)
target_link_libraries(std_sort tbb)

## Merge Sort
add_subdirectory(mergesort)

## Quick Sort
add_subdirectory(quicksort)

## GPU Radix Sort
add_subdirectory(radixsort-gpu)

## CPU Binary Search
add_subdirectory(searching-cpu)

## GPU Binary Search
add_subdirectory(searching-gpu)
