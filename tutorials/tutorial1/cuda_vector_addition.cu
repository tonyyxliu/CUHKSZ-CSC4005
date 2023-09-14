//
// Created by Liu Yuxuan on 2023/9/14.
// Email: 118010200@link.cuhk.edu.cn
//

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

void addVectors(const float* a, const float* b, float* c, int size) {
  for (int i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}

__global__ void addVectorsCUDA(const float* a, const float* b, float* c, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    c[tid] = a[tid] + b[tid];
  }
}

int main() {
  int size = 100000000;  // Size of the vectors
  int numBlocks[] = {1, 2, 4, 8, 16, 32, 64, 128};  // Different numbers of blocks to test

  // Allocate memory on the host
  float* h_a = new float[size];
  float* h_b = new float[size];
  float* h_c = new float[size];

  // Initialize the vectors
  for (int i = 0; i < size; i++) {
    h_a[i] = i;
    h_b[i] = i;
  }

  // Perform the vector addition on the CPU and record the time consumed
  auto cpuStartTime = std::chrono::high_resolution_clock::now();
  addVectors(h_a, h_b, h_c, size);
  auto cpuEndTime = std::chrono::high_resolution_clock::now();
  auto cpuDuration = std::chrono::duration_cast<std::chrono::microseconds>(cpuEndTime - cpuStartTime).count();

  // Print the result of the CPU computation
  std::cout << "CPU Execution Time: " << cpuDuration << " microseconds" << std::endl;

  // Allocate memory on the device
  float* d_a, * d_b, * d_c;
  cudaMalloc((void**)&d_a, size * sizeof(float));
  cudaMalloc((void**)&d_b, size * sizeof(float));
  cudaMalloc((void**)&d_c, size * sizeof(float));

  // Copy the input vectors from host to device
  cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice);

  // Perform the vector addition on the GPU for different numbers of blocks
  for (int i = 0; i < sizeof(numBlocks) / sizeof(numBlocks[0]); i++) {
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0); // 记录开始时间

    int blockSize = (int)ceil((float)size / numBlocks[i]);

    // Record the time consumed for the GPU computation
    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    // Launch the kernel with specified number of blocks and threads
    addVectorsCUDA<<<numBlocks[i], blockSize>>>(d_a, d_b, d_c, size);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    auto gpuEndTime = std::chrono::high_resolution_clock::now();
    auto gpuDuration = std::chrono::duration_cast<std::chrono::microseconds>(gpuEndTime - gpuStartTime).count();

    // Copy the result of the GPU computation from device to host
    cudaMemcpy(h_c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0); // 记录结束时间
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop); // 计算时间差

    // Print the result of the GPU computation
    // std::cout << "GPU Execution Time (Blocks: " << numBlocks[i] << "): " << gpuDuration << " microseconds" << std::endl;
    std::cout << "GPU Execution Time (Blocks: " << numBlocks[i] << "): " << elapsedTime << " milliseconds" << std::endl;

    // Calculate and display the speedup
    double speedup = cpuDuration / gpuDuration;
    std::cout << "Speedup (Blocks: " << numBlocks[i] << "): " << speedup << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  // Free memory on the device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  delete[] h_a;
  delete[] h_b;
  delete[] h_c;

  return 0;
}
