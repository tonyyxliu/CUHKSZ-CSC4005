//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Quick Sort
//

#include <iostream>
#include <vector>
#include "../utils.hpp"

int partition(std::vector<int> &vec, int low, int high) {
    /* Your code here!
       Implement parallel partition algorithm
    */
}

void quickSort(std::vector<int> &vec, int low, int high) {
    /* Your code here!
       Implement parallel quick sort with dynamic threads creation
    */
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

    auto start_time = std::chrono::high_resolution_clock::now();

    quickSort(vec, 0, size - 1);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Quick Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    checkSortResult(vec_clone, vec);

    return 0;
}