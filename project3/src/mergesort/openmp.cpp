//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Modified by Liu Yuxuan on 2024/10/26
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Task #4: Parallel Merge Sort with OpenMP
//

#include <iostream>
#include <vector>
#include "../utils.hpp"

/**
 * TODO: Implement parallel merge algorithm
 */
void merge(std::vector<int>& vec, int l, int m, int r) {
    /* Your codes here! */
}

/**
 * TODO: Implement parallel merge sort by dynamic threads creation
 */
void mergeSort(std::vector<int>& vec, int l, int r, int thread_num) {
    /* Your codes here! */
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable dist_type threads_num vector_size\n"
            );
    }
    const DistType dist_type = str_2_dist_type(std::string(argv[1]));
    const int thread_num = atoi(argv[2]);
    const int size = atoi(argv[3]);
    std::vector<int> vec = genRandomVec(size, dist_type); // use default seed
    std::vector<int> vec_clone = vec;

    std::vector<int> S(size);
    std::vector<int> L(size);
    std::vector<int> results(size);

    auto start_time = std::chrono::high_resolution_clock::now();

    mergeSort(vec, 0, size - 1, thread_num);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Merge Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    checkSortResult(vec_clone, vec);
    return 0;
}
