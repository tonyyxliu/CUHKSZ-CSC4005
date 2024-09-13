//
// Created by Liu Yuxuan on 2024/9/10
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Row-wise Pthread parallel implementation of smooth image filtering of JPEG
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <pthread.h>

#include "../utils.hpp"

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg NUM_THREADS\n";
        return -1;
    }
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    // Read input JPEG image
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    /**
     * TODO: Pthread PartC
     */
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
