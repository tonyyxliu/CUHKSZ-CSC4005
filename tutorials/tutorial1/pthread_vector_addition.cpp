#include <iostream>
#include <vector>
#include <chrono>
#include <pthread.h>

// Define the number of threads
const int numThreads = 4;
// Define vector size
const int vectorSize = 100000000;

// Struct to hold thread arguments
struct ThreadArgs {
  int startIndex;
  int endIndex;
  std::vector<int>* vectorA;
  std::vector<int>* vectorB;
  std::vector<int>* vectorC;
};

// Thread function for vector addition
void* vectorAddition(void* arg) {
  ThreadArgs* args = static_cast<ThreadArgs*>(arg);
  int startIndex = args->startIndex;
  int endIndex = args->endIndex;
  std::vector<int>* vectorA = args->vectorA;
  std::vector<int>* vectorB = args->vectorB;
  std::vector<int>* vectorC = args->vectorC;

  // Perform vector addition for the assigned range
  for (int i = startIndex; i < endIndex; ++i) {
    (*vectorC)[i] = (*vectorA)[i] + (*vectorB)[i];
  }

  return nullptr;
}

int main() {
  std::cout << "It's Pthread here!" << std::endl;

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

  // Create and join threads
  pthread_t threads[numThreads];
  ThreadArgs threadArgs[numThreads];
  int chunkSize = vectorSize / numThreads;
  for (int i = 0; i < numThreads; ++i) {
    threadArgs[i] = {i * chunkSize, (i + 1) * chunkSize, &vectorA, &vectorB, &vectorC};
    pthread_create(&threads[i], nullptr, vectorAddition, static_cast<void*>(&threadArgs[i]));
  }
  for (int i = 0; i < numThreads; ++i) {
    pthread_join(threads[i], nullptr);
  }

  // Stop timer
  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

  // Print the execution time
  std::cout << "Execution time: " << duration << " ms" << std::endl;

  return 0;
}
