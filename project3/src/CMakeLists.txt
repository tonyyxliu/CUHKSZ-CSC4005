## std::sort Sequential
add_executable(std_sort
        std_sort.cpp
        utils.cpp utils.hpp)
target_compile_options(std_sort PRIVATE -O2)

## Merge Sort
add_subdirectory(mergesort)

## Quick Sort
add_subdirectory(quicksort)

## Bucket Sort
add_subdirectory(bucketsort)

## PSRS Sorting
add_subdirectory(psrs)

## CPU Radix Sort
add_subdirectory(radixsort-cpu)
