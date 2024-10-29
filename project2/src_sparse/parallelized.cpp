#include <omp.h>
#include <mpi.h>
#include <stdexcept>
#include <chrono>
#include <iostream>
#include "sparse_matrix.hpp"

#define MASTER 0


/*****************************/
/**    Your code is here    **/
/*****************************/

SparseMatrix matrix_multiply(const SparseMatrix& A, const SparseMatrix& B) {
    if (A.n_col_ != B.n_row_) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }

    SparseMatrix C;

    /*****************************/
    /**    Your code is here    **/
    /*****************************/

    return C;
}


int main(int argc, char** argv) {
    if (argc != 5) {
        throw std::invalid_argument(
                "Invalid argument, should be: ./executable thread_num "
                "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
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

    int thread_num = atoi(argv[1]);
    omp_set_num_threads(thread_num);

    // Read Matrix
    const std::string matrix1_path = argv[2];

    const std::string matrix2_path = argv[3];

    const std::string result_path = argv[4];

    SparseMatrix matrix1, matrix2;

    matrix1.loadFromFile(matrix1_path);
    matrix2.loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();
    if (taskid == MASTER) {
        SparseMatrix result = matrix_multiply(matrix1, matrix2);

        // Your Code Here for Synchronization!

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time =
                std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                      start_time);

        result.save2File(result_path);

        std::cout << "Output file to: " << result_path << std::endl;

        std::cout << "Multiplication Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count()
                  << " milliseconds" << std::endl;
    } else {
        SparseMatrix result = matrix_multiply(matrix1, matrix2);

        // Your Code Here for Synchronization!
    }

    MPI_Finalize();
    return 0;
}
