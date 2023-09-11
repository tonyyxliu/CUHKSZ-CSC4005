// #include <iostream>
#include "stdio.h"
#include <cuda_runtime.h>

__global__ void helloKernel() {
    printf("Hello, World! from thread %d of block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    // Launch the kernel with 2 threads in a single block
    dim3 threadsPerBlock(2);
    dim3 numBlocks(1);
    helloKernel<<<numBlocks, threadsPerBlock>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    return 0;
}
