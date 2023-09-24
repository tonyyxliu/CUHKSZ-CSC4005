//
// Created by Liu Yuxuan on 2023/9/14.
// Email: 118010200@link.cuhk.edu.cn
//

#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

int main() {
  std::cout << "It's OpenMP here!" << std::endl;

  // Define vector size
  const int vectorSize = 100000000;

  // Create vectors
  std::vector<int> vectorA(vectorSize);
  std::vector<int> vectorB(vectorSize);
  std::vector<int> vectorC(vectorSize);

  // Initialize vectors
  for (int i = 0; i < vectorSize; ++i) {
    vectorA[i] = i;
    vectorB[i] = i;
  }

  // Start timer
  auto startTime = std::chrono::high_resolution_clock::now();

// Perform vector addition in parallel
#pragma omp parallel for
  for (int i = 0; i < vectorSize; ++i) {
    vectorC[i] = vectorA[i] + vectorB[i];
  }

  // Stop timer
  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

  // Print the execution time
  std::cout << "Execution time: " << duration << " ms" << std::endl;

  return 0;
}
