#include <iostream>

int main() {
    #pragma acc parallel num_gangs(4)
    {
        int thread_id = omp_get_thread_num();
        std::cout << "Hello from thread " << thread_id << std::endl;
    }

    return 0;
}
