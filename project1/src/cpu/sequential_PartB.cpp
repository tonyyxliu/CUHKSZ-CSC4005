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

    // Apply the filter to the image
    unsigned char* filteredImage =
        new unsigned char[input_jpeg.width * input_jpeg.height *
                          input_jpeg.num_channels];

    memset(filteredImage, 0,
           input_jpeg.width * input_jpeg.height * input_jpeg.num_channels);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int y = 1; y < input_jpeg.height - 1; y++)
    {
        for (int x = 1; x < input_jpeg.width - 1; x++)
        {
            int r_id = (y * input_jpeg.width + x) * input_jpeg.num_channels;
            int g_id = r_id + 1;
            int b_id = r_id + 2;

            float r_sum =
                linear_filter(input_jpeg.buffer, filter, r_id, input_jpeg.width,
                              input_jpeg.num_channels);

            float g_sum =
                linear_filter(input_jpeg.buffer, filter, g_id, input_jpeg.width,
                              input_jpeg.num_channels);

            float b_sum =
                linear_filter(input_jpeg.buffer, filter, b_id, input_jpeg.width,
                              input_jpeg.num_channels);

            filteredImage[r_id] = clamp_pixel_value(r_sum);
            filteredImage[g_id] = clamp_pixel_value(g_sum);
            filteredImage[b_id] = clamp_pixel_value(b_sum);
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