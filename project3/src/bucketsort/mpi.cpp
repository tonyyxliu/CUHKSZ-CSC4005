//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Modified by Liu Yuxuan on 2024/10/26
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Task #1: Parallel Bucket Sort with MPI
//

#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0

void insertionSort(std::vector<int> &bucket)
{
    /* You may print out the data size in each bucket here to see how severe the load imbalance is */
    for (int i = 1; i < bucket.size(); ++i)
    {
        int key = bucket[i];
        int j = i - 1;

        while (j >= 0 && bucket[j] > key)
        {
            bucket[j + 1] = bucket[j];
            j--;
        }

        bucket[j + 1] = key;
    }
}

/**
 * TODO: Parallel Bucket Sort with MPI
 * @param vec: input vector for sorting
 * @param num_buckets: number of buckets
 * @param numtasks: number of processes for sorting
 * @param taskid: the rank of the current process
 * @param status: MPI_Status for message passing
 */
void bucketSort(std::vector<int> &vec, int num_buckets, int numtasks, int taskid, MPI_Status *status)
{
    /* Your codes here! */
}

int main(int argc, char **argv)
{
    // Verify input argument format
    if (argc != 4)
    {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable dist_type vector_size bucket_num\n");
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

    const DistType dist_type = str_2_dist_type(std::string(argv[1]));
    const int size = atoi(argv[2]);
    const int bucket_num = atoi(argv[3]);
    std::vector<int> vec = genRandomVec(size, dist_type); // use default seed
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();

    bucketSort(vec, bucket_num, numtasks, taskid, &status);

    if (taskid == MASTER)
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);

        std::cout << "Bucket Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                  << std::endl;

        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}
