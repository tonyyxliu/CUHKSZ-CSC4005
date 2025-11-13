//
// Created by Fang Zihao on 2025/10/31.
// Email: zihaofang1@link.cuhk.edu.cn
//
// Parallel Quick Sort
//

#include <iostream>
#include <vector>
#include <thread>
#include <omp.h> 
#include "../utils.hpp"

int partition(std::vector<int> &vec, int low, int high) {
    /* Your code here!
        Implement partitioning
    */

}

void prefix_sum(std::vector<int> &vec, int low, int high) {
    /* Your code here!
       Implement prefix sum
    */
}

void prefix_sum_parallel(std::vector<int> &vec, int low, int high, int threads_num) {
    /* Your code here!
       Implement parallel prefix sum
    */
}

int partition_parallel(std::vector<int> &vec, std::vector<int> &results, std::vector<int> &S, std::vector<int> &L, int low, int high, int threads_num) {
    /* Your code here!
       Implement parallel partitioning with parallel prefix sum
    */
}

void quickSort(std::vector<int> &vec, std::vector<int> &results, std::vector<int> &S, std::vector<int> &L, int low, int high, int threads, int threads_limit) {
    /* Your code here!
       Implement quicksort
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
    std::vector<int> vec = createUniformVec(size); // use default seed
    std::vector<int> vec_clone = vec;

    std::vector<int> S(size);
    std::vector<int> L(size);
    std::vector<int> results(size);

    auto start_time = std::chrono::high_resolution_clock::now();

    quickSort(vec, results, S, L, 0, size - 1, 1, thread_num);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Quick Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    checkSortResult(vec_clone, vec);

    return 0;
}