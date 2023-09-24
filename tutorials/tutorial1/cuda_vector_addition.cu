//
// Created by Liu Yuxuan on 2023/9/14.
// Email: 118010200@link.cuhk.edu.cn
//

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

constexpr int VECTOR_SIZE = 100000000;  // Size of the vectors

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
  int numBlocks[] = {1, 2, 4, 8, 16, 32, 64, 128};  // Different numbers of blocks to test

  // Allocate memory on the host
  float* h_a = new float[VECTOR_SIZE];
  float* h_b = new float[VECTOR_SIZE];
  float* h_c = new float[VECTOR_SIZE];

  // Initialize the vectors
  for (int i = 0; i < VECTOR_SIZE; i++) {
    h_a[i] = i;
    h_b[i] = i;
  }

  // Perform the vector addition on the CPU and record the time consumed
  auto cpuStartTime = std::chrono::high_resolution_clock::now();
  addVectors(h_a, h_b, h_c, VECTOR_SIZE);
  auto cpuEndTime = std::chrono::high_resolution_clock::now();
  auto cpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(cpuEndTime - cpuStartTime).count();

  // Print the result of the CPU computation
  std::cout << "CPU Execution Time: " << cpuDuration << " milliseconds" << std::endl;

  // Allocate memory on the device
  float* d_a, * d_b, * d_c;
  cudaMalloc((void**)&d_a, VECTOR_SIZE * sizeof(float));
  cudaMalloc((void**)&d_b, VECTOR_SIZE * sizeof(float));
  cudaMalloc((void**)&d_c, VECTOR_SIZE * sizeof(float));

  // Copy the input vectors from host to device
  cudaMemcpy(d_a, h_a, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

  // Perform the vector addition on the GPU for different numbers of blocks
  for (int i = 0; i < sizeof(numBlocks) / sizeof(numBlocks[0]); i++) {
    cudaEvent_t start, stop;
    float gpuDuration;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0); // GPU start time

    int blockSize = (int)ceil((float)VECTOR_SIZE / numBlocks[i]);

    // Launch the kernel with specified number of blocks and threads
    addVectorsCUDA<<<numBlocks[i], blockSize>>>(d_a, d_b, d_c, VECTOR_SIZE);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy the result of the GPU computation from device to host
    cudaMemcpy(h_c, d_c, VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);

    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);
    std::cout << "GPU Execution Time (Blocks: " << numBlocks[i] << "): " << gpuDuration << " milliseconds" << std::endl;

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
