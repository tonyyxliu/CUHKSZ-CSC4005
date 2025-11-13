//
// Created by Lyu You on 2024/10/16
// Email: 121090404@link.cuhk.edu.cn
//
// Sequential Radix Sort
//

#include <openacc.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "../utils.hpp"

#define BASE 256
#define BASE_BITS 8
#define NUM_GANGS 1024

void radixSort(std::vector<int> &vec) {
    int n = vec.size();
    /* Your code here!
        Implement GPU Radix sort
    */
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 2) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size\n"
            );
    }

    const int size = atoi(argv[1]);

    const int seed = 4005;

    std::vector<int> vec = createUniformVec(size, seed);
    std::vector<int> vec_clone = vec;
    auto start_time = std::chrono::high_resolution_clock::now();

    radixSort(vec);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Radix Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;
    
    checkSortResult(vec_clone, vec);
    return 0;
}