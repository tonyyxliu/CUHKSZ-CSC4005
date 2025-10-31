//
// Created by Mengkang Li on 2025/10/27.
//
// Sequential Binary Search for Data Array
//

#include <iostream>
#include <vector>
#include "../utils.hpp"

// Binary Search - finds the FIRST occurrence of target
int binarySearch(const std::vector<int>& vec, int target) {
    int left = 0;
    int right = vec.size() - 1;
    int result = -1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (vec[mid] == target) {
            result = mid;
            right = mid - 1;  // Continue searching in left half for first occurrence
        } else if (vec[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return result;
}

std::vector<int> binarySearchArray(const std::vector<int>& vec, 
                                    const std::vector<int>& search_targets) {
    std::vector<int> results(search_targets.size());
    
    for (int i = 0; i < search_targets.size(); i++) {
        results[i] = binarySearch(vec, search_targets[i]);
    }
    
    return results;
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
    
    const int search_size = size / 10;  // 10% of array size
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
    
    std::cout << "Sequential Array Binary Search Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;
    
    checkSearchResult(vec, search_targets, results);
    
    return 0;
}

