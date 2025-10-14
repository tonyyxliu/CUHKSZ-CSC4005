//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Modified by Liu Yuxuan on 2025/10/12
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// OpenMp + AutoVec + Reordering Matrix Multiplication
//

#include <stdexcept>
#include <chrono>
#include <omp.h>
#include "matrix.hpp"

/**
 * Matmul with Loop Re-ordering, Serving as the utility function for blocks
 */
void matrix_multiply_loop_reorder(const Matrix& matrix1, const Matrix& matrix2,
                                  MAT_DATATYPE* __restrict__ result_data)
{
    if (matrix1.getCols() != matrix2.getRows())
    {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    /**
     * Refer to previous tasks
     */
}

/**
 * Multi-Threaded Matmul
 */
Matrix matrix_multiply_openmp(const Matrix& matrix1, const Matrix& matrix2,
                              int num_threads, size_t block_size)
{
    if (matrix1.getCols() != matrix2.getRows())
    {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
    std::cout << "M = " << M << ", N = " << N << ", K = " << K << std::endl;

    Matrix result(M, N);

    /**
     * TODO: build upton previous tasks
     */

    return result;
}

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable thread_num block_size"
            "/path/to/matrix1 /path/to/matrix2\n");
    }

    const int num_threads = std::atoi(argv[1]);
    const size_t block_size = static_cast<size_t>(std::atoi(argv[2]));
    const std::string matrix1_path = argv[3];
    const std::string matrix2_path = argv[4];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);
    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result =
        matrix_multiply_openmp(matrix1, matrix2, num_threads, block_size);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    Matrix ground_truth = Matrix::getResultMatrix(matrix1_path, matrix2_path);
    std::cout << "Verification: "
              << ((Matrix::isIdentical(result, ground_truth)) ? "Passed"
                                                              : "Failed")
              << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;
    return 0;
}