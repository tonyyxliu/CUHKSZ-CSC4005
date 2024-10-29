//
// Created by Lyu You on 2024/10/17
// Email: 121090404@link.cuhk.edu.cn
//
// Task #6 (Extra Credits): Parallel Radix Sort with OpenMP
//

#include <iostream>
#include <vector>

#include <omp.h> 

#include "../utils.hpp"

#define BASE 16384

/**
 * TODO: Parallel Radix Sort using OpenMP
 */
void radixSort(std::vector<int> &vec, int threads_num) {
    /* Your codes here */
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable threads_num vector_size\n"
            );
    }

    const int thread_num = atoi(argv[1]);

    const int size = atoi(argv[2]);

    const int seed = 4005;

    std::vector<int> vec = createUniformVec(size, seed);
    std::vector<int> vec_clone = vec;

    omp_set_num_threads(thread_num);
    
    auto start_time = std::chrono::high_resolution_clock::now();

    radixSort(vec, thread_num);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Radix Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    checkSortResult(vec_clone, vec);

    return 0;
}
