#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>
#include <ctime>

__host__ __device__ float generate_random(unsigned int seed, int global_idx) {
    unsigned int z = seed + global_idx;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    z = z ^ (z >> 31);

    return (float)(z & 0xFFFFFF) / 16777216.0f;
}

__global__ void softmax_forward_kernel(float* out, const float* inp, const float* mask, 
                                       int N, int C, 
                                       float scale, float dropout_p, unsigned int seed) {
    // Your code here
}

void cpu_softmax(float* out, const float* inp, const float* mask, 
                 int N, int C, float scale, float dropout_p, unsigned int seed) {
    for (int i = 0; i < N; ++i) {
        float max_val = -INFINITY;
        for (int j = 0; j < C; ++j) {
            float val = inp[i * C + j] * scale;
            if (mask[i * C + j] < 0.5f) val = -INFINITY;
            if (val > max_val) max_val = val;
        }

        float sum = 0.0f;
        for (int j = 0; j < C; ++j) {
            float val = inp[i * C + j] * scale;
            if (mask[i * C + j] < 0.5f) {
                out[i * C + j] = 0.0f;
            } else {
                float res = expf(val - max_val);
                out[i * C + j] = res;
                sum += res;
            }
        }

        // Normalize & Dropout
        float dropout_scale = 1.0f / (1.0f - dropout_p);
        for (int j = 0; j < C; ++j) {
            float val = out[i * C + j] / sum;
            
            if (dropout_p > 0.0f) {
                int global_idx = i * C + j;
                float rand_val = generate_random(seed, global_idx);
                if (rand_val < dropout_p) {
                    val = 0.0f;
                } else {
                    val *= dropout_scale;
                }
            }
            out[i * C + j] = val;
        }
    }
}

bool verify(float* gpu_res, float* cpu_res, int N, int C) {
    float max_diff = 0.0f;
    for (int i = 0; i < N * C; ++i) {
        float diff = std::abs(gpu_res[i] - cpu_res[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "Max absolute difference: " << max_diff << std::endl;
    return max_diff < 1e-4;
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: ./softmax N C\n";
        return -1;
    }

    int N = std::atoi(argv[1]);
    int C = std::atoi(argv[2]);
    
    float scale = 1.0f / sqrt(float(C));
    float dropout_p = 0.1f;
    unsigned int seed = 12345;

    size_t size_bytes = N * C * sizeof(float);
    float* h_input = new float[N * C];
    float* h_mask = new float[N * C];
    float* h_output_gpu = new float[N * C];
    float* h_output_cpu = new float[N * C];

    srand(time(0));
    for (int i = 0; i < N * C; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
        h_mask[i] = (static_cast<float>(rand()) / RAND_MAX > 0.3f) ? 1.0f : 0.0f;
    }

    float *d_input, *d_output, *d_mask;
    cudaMalloc(&d_input, size_bytes);
    cudaMalloc(&d_output, size_bytes);
    cudaMalloc(&d_mask, size_bytes);

    cudaMemcpy(d_input, h_input, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, size_bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(256);
    dim3 gridDim(N);
    size_t shared_mem_size = blockDim.x * sizeof(float);

    // Correctness
    std::cout << "Running Correctness Check..." << std::endl;
    
    softmax_forward_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        d_output, d_input, d_mask, N, C, scale, dropout_p, seed
    );
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_gpu, d_output, size_bytes, cudaMemcpyDeviceToHost);

    cpu_softmax(h_output_cpu, h_input, h_mask, N, C, scale, dropout_p, seed);

    if (verify(h_output_gpu, h_output_cpu, N, C)) {
        std::cout << "PASS: GPU result matches CPU result." << std::endl;
    } else {
        std::cout << "FAIL: Mismatch detected!" << std::endl;
    }

    // Benchmark
    std::cout << "\nRunning Benchmark (" << std::endl;

    // Warm up
    softmax_forward_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        d_output, d_input, d_mask, N, C, scale, dropout_p, seed
    );
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int run_times = 10;
    float total_time = 0.0f;  
    float single_time = 0.0f;

    for (int i = 0; i < run_times; ++i) {
        cudaEventRecord(start);
        softmax_forward_kernel<<<gridDim, blockDim, shared_mem_size>>>(
            d_output, d_input, d_mask, N, C, scale, dropout_p, seed
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&single_time, start, stop);
        total_time += single_time;
    }

    float avg_duration = total_time / run_times;

    std::cout << "GPU Execution Time (10 runs): " << std::endl;
    std::cout << "Total Time: " << total_time << " ms" << std::endl;
    std::cout << "Average Time: " << avg_duration << " ms" << std::endl;
    delete[] h_input;
    delete[] h_mask;
    delete[] h_output_gpu;
    delete[] h_output_cpu;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}