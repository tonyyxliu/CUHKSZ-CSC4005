#include <iostream>
#include <omp.h>

int main() {
    int num_threads = 4; // Set the desired number of threads

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        std::cout << "Hello from thread " << thread_id << std::endl;
    }

    return 0;
}
