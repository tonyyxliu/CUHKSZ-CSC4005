//
// Created by Mengkang Li on 2025/10/27.
//
// Parallel Binary Search for Data Array on CPU
//

#include <iostream>
#include <vector>
#include "../utils.hpp"

// Binary Search - finds the FIRST occurrence of targets from [i, i + BATCH_SIZE - 1] in range [0, size - 1]
#pragma acc routine seq
void binarySearch(const int* vec, int size, int bits, const int* targets, int i, int* results) {
    /* Your code here!
    Optimized GPU binary search
    */
}

std::vector<int> binarySearchArray(const std::vector<int>& vec, 
                                    const std::vector<int>& search_targets) {
    int n = vec.size();
    int nbits = 31 - __builtin_clz(n);
    int search_size = search_targets.size();
    std::vector<int> results(search_size);
    /* Your code here
       Binary search for array
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
    
    std::vector<int> vec = createUniformVec(size);
    std::sort(vec.begin(), vec.end());
    
    const int search_size = size / 10;
    std::vector<int> search_targets(search_size);
    
    std::mt19937 gen(CSC4005_SEED);
    std::uniform_int_distribution<> dis(0, size - 1);
    
    for (int i = 0; i < search_size; i++) {
        int idx = dis(gen);
        search_targets[i] = vec[idx];
    }
    std::sort(search_targets.begin(), search_targets.end());
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<int> results = binarySearchArray(vec, search_targets);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Parallel Array Binary Search Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;
    
    checkSearchResult(vec, search_targets, results);
    
    return 0;
}

