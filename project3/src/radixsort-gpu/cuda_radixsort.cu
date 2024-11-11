//
// Created by Yuan Xu on 2024/10/19.
// Email: yuanxu1@link.cuhk.edu.cn
//
// Template for counting sort (radix = 1) / radix sort (recommend radix = 256)
// Execution command: srun -n 1 --gpus 1 ./src/gpu/cuda_radixsort 100000000
//

#include <iostream>

#include <cuda_runtime.h> // CUDA Header

#include "../utils.hpp"


void radixSortGPU(unsigned int* inp, unsigned int* cnt, unsigned int* res, int n, int b) {
    // Implement your GPU implementation for radix sort here
    // NOTE: You CAN modify the structure of this script to achieve your goal
    // TODO: implement radix sort on GPU
    
}

unsigned int inp[1<<27], cnt[1<<27], res[1<<27];

/* Radix Sort
CPU version:
for i = 0 to n-1 : c[a[i]+1] += 1
for i = 1 to m-1 : c[i] += c[i-1]
for i = 0 to n-1 : b[c[a[i]]++] = a[i] 
*/
int main(int argc, char** argv)
{
    const int size = atoi(argv[1]); 
    const int seed = 4005;
    int n = 1, b = 0;
    while (n < size) { 
        n <<= 1; 
        b += 1; 
    }
    std::vector<int> vec = createRandomVec(size, seed);
    std::vector<int> vec_clone = vec;
    for (int i = 0; i < size; i++) inp[i] = vec[i];
    memset(inp + size, 0, (n-size) * sizeof(unsigned int));
    memset(cnt, 0, n * sizeof(unsigned int));
    // Allocate memory on device (GPU)
    unsigned int* d_inp;
    unsigned int* d_cnt;
    unsigned int* d_res;
    cudaMalloc((void**)&d_inp, n * sizeof(unsigned int));
    cudaMalloc((void**)&d_cnt, n * sizeof(unsigned int));
    cudaMalloc((void**)&d_res, n * sizeof(unsigned int));
    // Copy input data from host to device
    cudaMemcpy(d_inp, inp, n * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cnt, cnt, n * sizeof(unsigned int), cudaMemcpyHostToDevice);
    // Create CUDA timer
    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0); // GPU start time
    // Launch the device computation threads!
    radixSortGPU(d_inp, d_cnt, d_res, n, b);
    // Print the result of the GPU computation
    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuDuration, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // Copy output data from device to host
    cudaMemcpy(res, d_res, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_inp);
    cudaFree(d_cnt);
    cudaFree(d_res);
    for (int i = 0; i < size; i++) vec[i] = res[i+n-size];
    std::cout << "GPU Radix Sort Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds" << std::endl;
   
    checkSortResult(vec_clone, vec);
    return 0;
}
