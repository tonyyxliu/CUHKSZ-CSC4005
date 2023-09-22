//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// CUDA implementation of transforming a JPEG image from RGB to gray
//

#include <iostream>

#include <cuda_runtime.h> // CUDA Header

#include "utils.hpp"

// CUDA kernel functon：RGB to Gray
__global__ void rgbToGray(const unsigned char* input, unsigned char* output,
                          int width, int height, int num_channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height)
    {
        unsigned char r = input[idx * num_channels];
        unsigned char g = input[idx * num_channels + 1];
        unsigned char b = input[idx * num_channels + 2];
        output[idx] =
            static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
    }
}

int main(int argc, char** argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read from input JPEG
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    // Allocate memory on host (CPU)
    auto grayImage = new unsigned char[input_jpeg.width * input_jpeg.height];   // num_channels = 1
    // Allocate memory on device (GPU)
    unsigned char* d_input;
    unsigned char* d_output;
    cudaMalloc((void**)&d_input, input_jpeg.width * input_jpeg.height *
                                     input_jpeg.num_channels *
                                     sizeof(unsigned char));
    cudaMalloc((void**)&d_output,
               input_jpeg.width * input_jpeg.height * sizeof(unsigned char));
    // Copy input data from host to device
    cudaMemcpy(d_input, input_jpeg.buffer,
               input_jpeg.width * input_jpeg.height * input_jpeg.num_channels *
                   sizeof(unsigned char),
               cudaMemcpyHostToDevice);
    // Computation: RGB to Gray
    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int blockSize = 512; // 256
    // int numBlocks =
    //     (input_jpeg.width * input_jpeg.height + blockSize - 1) / blockSize;
    int numBlocks = (input_jpeg.width * input_jpeg.height) / blockSize + 1;
    cudaEventRecord(start, 0); // GPU start time
    rgbToGray<<<numBlocks, blockSize>>>(d_input, d_output, input_jpeg.width,
                                        input_jpeg.height,
                                        input_jpeg.num_channels);
    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);
    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);
    // Copy output data from device to host
    cudaMemcpy(grayImage, d_output,
               input_jpeg.width * input_jpeg.height * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);
    // Write GrayImage to output JPEG
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{grayImage, input_jpeg.width, input_jpeg.height, 1,
                         JCS_GRAYSCALE};
    if (write_to_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Release allocated memory on device and host
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] input_jpeg.buffer;
    delete[] grayImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}