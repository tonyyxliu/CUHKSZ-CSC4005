//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Merge Sort
//

#include <iostream>
#include <vector>
#include "../utils.hpp"

void insertionSort(std::vector<int>& vec, int low, int high) {
    for (int i = low+1; i <= high; ++i) {
        int key = vec[i], j = i - 1;
        while (j >= low && vec[j] > key) {
            vec[j + 1] = vec[j];
            j--;
        }
        vec[j + 1] = key;
    }
}

void merge(std::vector<int>& vec, int l, int m, int r) {
    /* Your code here!
       Implement parallel merge algorithm
    */
    
}

void parMerge(std::vector<int>& vec, std::vector<int>& vec2, int l1, int r1, int l2, int r2, int l3, int depth) {
    /* Your code here!
       Implement parallel merge algorithm
    */
    
}

void parMergeSort(std::vector<int>& vec, const int &l, const int &r, int depth) {
    /* Your code here!
       Implement parallel merge sort algorithm
    */
}

void mergeSort(std::vector<int>& vec, int l, int r, int thread_num) {
    /* Your code here!
       Implement parallel merge sort by dynamic threads creation
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

    mergeSort(vec, 0, size - 1, thread_num);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Merge Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    checkSortResult(vec_clone, vec);
}
