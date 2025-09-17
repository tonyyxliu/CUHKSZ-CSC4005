//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// Sequential implementation of converting a JPEG from RGB to gray

#include <memory.h>

#include <chrono>
#include <cmath>
#include <iostream>

#include "../utils.hpp"

inline ColorValue linear_filter_vectorize(
    const ColorValue* const __restrict__ values,
    const float (&filter)[FILTERSIZE][FILTERSIZE], const int pixel_id,
    const int width, const int num_channels)
{
    float sum = 0.0f;
    const int line_width = width * num_channels;
    // sum += values[pixel_id] * filter[1][1];
    // sum += values[pixel_id - num_channels] * filter[1][0];
    // sum += values[pixel_id + num_channels] * filter[1][2];
    // sum += values[pixel_id - line_width] * filter[0][1];
    // sum += values[pixel_id - line_width - num_channels] * filter[0][0];
    // sum += values[pixel_id - line_width + num_channels] * filter[0][2];
    // sum += values[pixel_id + line_width] * filter[2][1];
    // sum += values[pixel_id + line_width - num_channels] * filter[2][0];
    // sum += values[pixel_id + line_width + num_channels] * filter[2][2];

    // Read all pixel values using index array
    const int indices[9] = {pixel_id - line_width - num_channels,
                            pixel_id - line_width,
                            pixel_id - line_width + num_channels,
                            pixel_id - num_channels,
                            pixel_id,
                            pixel_id + num_channels,
                            pixel_id + line_width - num_channels,
                            pixel_id + line_width,
                            pixel_id + line_width + num_channels};
    ColorValue neighbor_values[9];
#pragma GCC unroll 9
    for (int i = 0; i < 9; i++)
    {
        neighbor_values[i] = values[indices[i]];
    }

    // weights of filter matrix
    const float weight[9] = {filter[0][0], filter[0][1], filter[0][2],
                             filter[1][0], filter[1][1], filter[1][2],
                             filter[2][0], filter[2][1], filter[2][2]};

    // Compute the sum
#pragma GCC ivdep
#pragma GCC unroll 9
    for (int i = 0; i < 9; i++)
    {
        sum += neighbor_values[i] * weight[i];
    }

    return clamp_pixel_value(sum);
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read JPEG File
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg = read_from_jpeg(input_filename);

    // Allocate memory for filtered image
    const int width = input_jpeg.width;
    const int height = input_jpeg.height;
    const int num_channels = input_jpeg.num_channels;
    auto filteredImage = new ColorValue[width * height * num_channels];
    memset(filteredImage, 0, width * height * num_channels);

    auto start_time = std::chrono::high_resolution_clock::now();

    /* Pixels in the boundary can be ignored in this assignment */
    for (int y = 1; y < height - 1; y++)
    {
#pragma omp simd
        for (int x = 1; x < width - 1; x++)
        {
            int r_idx = (y * width + x) * num_channels;
            int g_idx = r_idx + 1;
            int b_idx = r_idx + 2;

            filteredImage[r_idx] = linear_filter_vectorize(
                input_jpeg.buffer, filter, r_idx, width, num_channels);

            filteredImage[g_idx] = linear_filter_vectorize(
                input_jpeg.buffer, filter, g_idx, width, num_channels);

            filteredImage[b_idx] = linear_filter_vectorize(
                input_jpeg.buffer, filter, b_idx, width, num_channels);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

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
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
