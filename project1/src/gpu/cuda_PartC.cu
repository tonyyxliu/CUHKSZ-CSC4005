//
// Created by Liu Yuxuan on 2024/9/11
// Modified from Zhong Yebin's PartB on 2023/9/16
//
// Email: yebinzhong@link.cuhk.edu.cn
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// CUDA implementation of bilateral filtering on JPEG image
//

#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#include "../utils.hpp"

/**
 * Demo kernel device function to clamp pixel value
 * 
 * You may mimic this to implement your own kernel device functions
 */
__device__ unsigned char d_clamp_pixel_value(float value)
{
    return value > 255 ? 255
           : value < 0 ? 0
                       : static_cast<unsigned char>(value);
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read input JPEG image in structure-of-array form
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    /**
     * TODO: CUDA PartC
     */
    return 0;
}
