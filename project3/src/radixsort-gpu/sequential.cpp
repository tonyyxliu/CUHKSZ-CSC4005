//
// Created by Lyu You on 2024/10/16
// Email: 121090404@link.cuhk.edu.cn
//
// Sequential Radix Sort
//

#include <iostream>
#include <vector>
#include "../utils.hpp"

#define BASE 256
#define BASE_BITS 8

void radixSort(std::vector<int> &vec) {
    int n = vec.size();

    // Get the pointer of element
    int *vec_raw = vec.data();

    // Counting sort for each digit
    int *output = new int[n];
    int count[BASE];
    int start_pos[BASE];
    int offset[BASE];

    for (int shift = 0; shift < 32; shift += BASE_BITS) {
        
        memset(count, 0, sizeof(int) * BASE);
        for (int i = 0; i < n; i++)
            count[(vec_raw[i] >> shift) & (BASE - 1)]++;

        memset(start_pos, 0, sizeof(int) * BASE);
        for (int d = 1; d < BASE; d++) {
            start_pos[d] = start_pos[d - 1] + count[d - 1];
        }

        memset(offset, 0, sizeof(int) * BASE);
        for (int i = 0; i < n; i++) {
            int digit = (vec_raw[i] >> shift) & (BASE - 1);
            int pos = start_pos[digit] + offset[digit];
            output[pos] = vec_raw[i];
            offset[digit]++;
        }

        // Assign elements back to the input vector
        memcpy(vec_raw, output, sizeof(int) * n);
    }
    delete[] output;
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