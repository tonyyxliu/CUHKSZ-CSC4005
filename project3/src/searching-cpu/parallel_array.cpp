//
// Created by Mengkang Li on 2025/10/27.
//
// Modified by Liu Yuxuan on 2024/10/28
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Task #4: Parallel Binary Search for Data Array on CPU
//

#include <iostream>
#include <vector>
#include <omp.h>
#include "../utils.hpp"

/**
 * TODO: Implement parallel binary search for multiple targets using OpenMP
 */
std::vector<int> binarySearchArray(const std::vector<int>& vec, 
                                    const std::vector<int>& search_targets,
                                    int thread_num) {
    /* Your codes here! */
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
    
    // Create and sort the array
    std::vector<int> vec = createUniformVec(size);
    std::sort(vec.begin(), vec.end());
    
    // Generate search targets (10% of array size)
    const int search_size = size / 10;
    std::vector<int> search_targets(search_size);
    
    std::mt19937 gen(CSC4005_SEED);
    std::uniform_int_distribution<> dis(0, size - 1);
    
    // Randomly select search targets from the sorted array
    for (int i = 0; i < search_size; i++) {
        int idx = dis(gen);
        search_targets[i] = vec[idx];
    }
    // Sort search targets to exploit locality (optional optimization)
    std::sort(search_targets.begin(), search_targets.end());
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<int> results = binarySearchArray(vec, search_targets, thread_num);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Parallel Array Binary Search Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;
    
    checkSearchResult(vec, search_targets, results);
    
    return 0;
}

