//
// Created by Lyu You on 2024/10/16
// Email: 121090404@link.cuhk.edu.cn
//
// Task #6 (Extra Credits): Sequential Radix Sort
//

#include <iostream>
#include <vector>

#include "../utils.hpp"

#define BASE 256

void radixSort(std::vector<int> &vec) {
    int n = vec.size();
    int i;
    long exp;

    // Get the largest element
    int max_element = vec[0];

    for (i = 0; i < n; ++i) {
        if (vec[i] > max_element) {
            max_element = vec[i];
        }
    }

    // Counting sort for each digit
    for (exp = 1; max_element / exp > 0; exp *= BASE) {
        std::vector<int> output(n);
        std::vector<int> count(BASE, 0);

        for (i = 0; i < n; i++)
            count[(vec[i] / exp) % BASE]++;

        std::vector<int> start_pos(BASE, 0);
        for (int d = 1; d < BASE; d++) {
            start_pos[d] = start_pos[d - 1] + count[d - 1];
        }

        std::vector<int> offset(BASE, 0);
        for (int i = 0; i < n; i++) {
            int digit = (vec[i] / exp) % BASE;
            int pos = start_pos[digit] + offset[digit];
            output[pos] = vec[i];
            offset[digit]++;
        }

        // Assign elements back to the input vector
        for (i = 0; i < n; i++) {
            vec[i] = output[i];
        }
    }

    return;
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
