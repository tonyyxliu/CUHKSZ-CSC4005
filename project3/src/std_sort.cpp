//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Sequential Quick Sort
//

#include <iostream>
#include <string>
#include <vector>
#include "utils.hpp"

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable dist_type vector_size\n"
            );
    }
    const DistType dist_type = str_2_dist_type(std::string(argv[1]));
    const int size = atoi(argv[2]);
    std::vector<int> vec = genRandomVec(size, dist_type); // use default seed
    auto start_time = std::chrono::high_resolution_clock::now();
    std::sort(vec.begin(), vec.end());
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cout << "std::sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;
    return 0;
}
