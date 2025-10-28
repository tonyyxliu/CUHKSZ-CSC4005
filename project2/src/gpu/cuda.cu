//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Matrix Multiplication with CUDA, for bonus
//
#include <iostream>
#include <matrix_lowPre.hpp>
#include <chrono>
#include <cuda_runtime.h>
#include <kernels.cuh>

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable kernel_name"
            "/path/to/matrix1 /path/to/matrix2\n");
    }

    char *kernel_name = argv[1];
    const std::string matrix1_path = argv[2];
    const std::string matrix2_path = argv[3];
    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);
    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    if (matrix1.getCols() != matrix2.getRows())
    {
        std::cout << "Matrix size does not match!" << std::endl;
        return 0;
    }
    int M = matrix1.getRows(), N = matrix1.getCols(), K = matrix2.getCols();
    std::cout << "M = " << M << " N = " << N << " K = " << K << std::endl;
    Matrix result(M, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    MAT_DATATYPE *dMatrix1, *dMatrix2, *dMatrixr;
    size_t sizeMat1 = M * K * sizeof(MAT_DATATYPE);
    size_t sizeMat2 = K * N * sizeof(MAT_DATATYPE);
    size_t sizeMatr = M * N * sizeof(MAT_DATATYPE);
    cudaMalloc(&dMatrix1, sizeMat1);
    cudaMalloc(&dMatrix2, sizeMat2);
    cudaMalloc(&dMatrixr, sizeMatr);

    cudaMemcpy(dMatrix1, matrix1.getDataConst(), sizeMat1,
               cudaMemcpyHostToDevice);
    cudaMemcpy(dMatrix2, matrix2.getDataConst(), sizeMat2,
               cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    auto start_time = std::chrono::high_resolution_clock::now();

    cudaEventRecord(start);

    launch_kernel(kernel_name, dMatrix1, dMatrix2, dMatrixr, M, N, K);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    MAT_DATATYPE *result_data = result.getData();
    cudaMemcpy(result_data, dMatrixr, sizeMatr, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    Matrix ground_truth = Matrix::getResultMatrix(matrix1_path, matrix2_path);
    std::cout << "Verification: "
              << ((Matrix::isIdentical(result, ground_truth)) ? "Passed"
                                                              : "Failed")
              << std::endl;
    std::cout << "Execution Time: " << milliseconds << " milliseconds"
              << std::endl;

    double GFLOP = (M / 1024) * (N / 1024) * (K / 1024) * 2;
    std::cout << "GFLOPS: " << GFLOP / milliseconds * 1000 << std::endl;

    cudaFree(dMatrix1);
    cudaFree(dMatrix2);
    cudaFree(dMatrixr);
}