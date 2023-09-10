#include <iostream>
#include <pthread.h>
#include <cstdlib>

#define NUM_THREADS 4

void* sayHello(void* threadId) {
    long tid = (long)threadId;
    std::cout << "Hello from thread " << tid << std::endl;
    pthread_exit(NULL);
}

int main() {
    pthread_t threads[NUM_THREADS];
    int rc;
    long t;

    for (t = 0; t < NUM_THREADS; t++) {
        std::cout << "Creating thread " << t << std::endl;
        rc = pthread_create(&threads[t], NULL, sayHello, (void*)t);
        if (rc) {
            std::cerr << "Error: Unable to create thread, " << rc << std::endl;
            exit(-1);
        }
    }

    // Wait for each thread to complete
    for (t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    return 0;
}
