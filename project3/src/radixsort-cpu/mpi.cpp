//
// Created by Lyu You on 2024/10/17
// Email: 121090404@link.cuhk.edu.cn
//
// Task #6 (Extra Credits): Parallel Radix Sort with MPI
//

#include <iostream>
#include <vector>
#include <queue>
#include <tuple>

#include <mpi.h>

#include "../utils.hpp"

#define MASTER 0
#define BASE 256

/**
 * TODO: Parallel Radix Sort using MPI
 */
void radixSort(std::vector<int>& vec, int numtasks, int taskid, MPI_Status* status) {
    /* Your codes here */
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 2) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size\n"
            );
    }
    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    const int size = atoi(argv[1]);

    const int seed = 4005;

    std::vector<int> vec = createUniformVec(size, seed);
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();

    radixSort(vec, numtasks, taskid, &status);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Radix Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;

        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}
