//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// CUDA implementation of image filtering on JPEG
//

#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include "../utils.hpp"

__device__ float d_linear_filter(unsigned char* image_buffer,
                                 const float (*d_filter)[FILTERSIZE],
                                 int pixel_id, int width, int num_channels)
{
    float sum = 0;
    int line_width = width * num_channels;
    sum += image_buffer[pixel_id] * d_filter[1][1];
    sum += image_buffer[pixel_id - num_channels] * d_filter[1][0];
    sum += image_buffer[pixel_id + num_channels] * d_filter[1][2];
    sum += image_buffer[pixel_id - line_width] * d_filter[0][1];
    sum += image_buffer[pixel_id - line_width - num_channels] * d_filter[0][0];
    sum += image_buffer[pixel_id - line_width + num_channels] * d_filter[0][2];
    sum += image_buffer[pixel_id + line_width] * d_filter[2][1];
    sum += image_buffer[pixel_id + line_width - num_channels] * d_filter[2][0];
    sum += image_buffer[pixel_id + line_width + num_channels] * d_filter[2][2];
    return sum;
}

__device__ unsigned char d_clamp_pixel_value(float pixel)
{
    return pixel > 255 ? 255
           : pixel < 0 ? 0
                       : static_cast<unsigned char>(pixel);
}

__global__ void apply_filter_kernel(unsigned char* input_buffer,
                                    unsigned char* filtered_image, int width,
                                    int height, int num_channels,
                                    const float (*d_filter)[FILTERSIZE])
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1)
    {
        int r_id = (y * width + x) * num_channels;
        int g_id = r_id + 1;
        int b_id = r_id + 2;

        float r_sum =
            d_linear_filter(input_buffer, d_filter, r_id, width, num_channels);
        float g_sum =
            d_linear_filter(input_buffer, d_filter, g_id, width, num_channels);
        float b_sum =
            d_linear_filter(input_buffer, d_filter, b_id, width, num_channels);

        filtered_image[r_id] = d_clamp_pixel_value(r_sum);
        filtered_image[g_id] = d_clamp_pixel_value(g_sum);
        filtered_image[b_id] = d_clamp_pixel_value(b_sum);
    }
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // 读取JPEG图像
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg = read_from_jpeg(input_filename);
    // Apply the filter to the image
    size_t buffer_size =
        input_jpeg.width * input_jpeg.height * input_jpeg.num_channels;
    unsigned char* filteredImage = new unsigned char[buffer_size];

    // Allocate GPU memory
    unsigned char* d_input_buffer;
    unsigned char* d_filtered_image;
    float(*d_filter)[FILTERSIZE];

    cudaMalloc((void**)&d_input_buffer, buffer_size);
    cudaMalloc((void**)&d_filtered_image, buffer_size);
    cudaMalloc((void**)&d_filter, FILTERSIZE * FILTERSIZE * sizeof(float));

    cudaMemset(d_filtered_image, 0, buffer_size);

    // Copy input data from host to device
    cudaMemcpy(d_input_buffer, input_jpeg.buffer, buffer_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, FILTERSIZE * FILTERSIZE * sizeof(float),
               cudaMemcpyHostToDevice);

    // Set CUDA grid and block sizes
    dim3 blockDim(32, 32);
    dim3 gridDim((input_jpeg.width + blockDim.x - 1) / blockDim.x,
                 (input_jpeg.height + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Perform filtering on GPU
    cudaEventRecord(start, 0); // GPU start time
    // Launch CUDA kernel
    apply_filter_kernel<<<gridDim, blockDim>>>(
        d_input_buffer, d_filtered_image, input_jpeg.width, input_jpeg.height,
        input_jpeg.num_channels, d_filter);
    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);
    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);
    // Copy output data from GPU
    cudaMemcpy(filteredImage, d_filtered_image, buffer_size,
               cudaMemcpyDeviceToHost);

    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height,
                         input_jpeg.num_channels, input_jpeg.color_space};
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Post-processing
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    // Release GPU memory
    cudaFree(d_input_buffer);
    cudaFree(d_filtered_image);
    cudaFree(d_filter);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds"
              << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
