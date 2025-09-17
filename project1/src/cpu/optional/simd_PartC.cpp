//
// Created by Liu Yuxuan on 2024/9/10
// Modified on Yang Yufan's simd_PartB.cpp on 2023/9/16
// Email: yufanyang1@link.cuhk.edu.cn
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// SIMD (AVX2) implementation of transferring a JPEG picture from RGB to gray
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>

#include "../../utils.hpp"

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read JPEG File
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filepath);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    /**
     * TODO: SIMD PartC
     */
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
