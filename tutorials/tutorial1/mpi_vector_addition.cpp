//
// Created by Liu Yuxuan on 2023/9/14.
// Email: 118010200@link.cuhk.edu.cn
//

#include <iostream>
#include <mpi.h>
#include <vector>
#include <chrono>

int main(int argc, char** argv) {
  std::cout << "It's MPI here!" << std::endl;

  MPI_Init(NULL, NULL);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Define vector size
  const int vectorSize = 100000000;
  const int elementsPerProcess = vectorSize / size;

  // Create local vectors
  std::vector<int> localVectorA(elementsPerProcess);
  std::vector<int> localVectorB(elementsPerProcess);
  std::vector<int> localVectorC(elementsPerProcess);

  // Initialize local vectors
  for (int i = 0; i < elementsPerProcess; ++i) {
    localVectorA[i] = rank * elementsPerProcess + i;
    localVectorB[i] = (size - rank - 1) * elementsPerProcess + i;
  }

  // Start timer
  MPI_Barrier(MPI_COMM_WORLD);
  auto startTime = std::chrono::high_resolution_clock::now();

  // Perform vector addition
  for (int i = 0; i < elementsPerProcess; ++i) {
    localVectorC[i] = localVectorA[i] + localVectorB[i];
  }

  // Stop timer
  MPI_Barrier(MPI_COMM_WORLD);
  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

  // // Gather results from all processes
  // std::vector<int> finalVector(vectorSize);
  // MPI_Gather(localVectorC.data(), elementsPerProcess, MPI_INT, finalVector.data(), elementsPerProcess, MPI_INT, 0, MPI_COMM_WORLD);

  // Print the result from the root process
  if (rank == 0) {
    std::cout << "Execution time: " << duration << " ms" << std::endl;
  }

  MPI_Finalize();
  return 0;
}

