//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// Sequential implementation of converting a JPEG from RGB to gray
// (Array-of-Structure)
//

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
    JpegAOS input_jpeg = read_jpeg_aos(input_filename);
    if (input_jpeg.pixels == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    // Apply the filter to the image
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    Pixel* output_pixels = new Pixel[width * height];
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int channel = 0; channel < num_channels; ++channel)
    {
        for (int row = 1; row < height - 1; ++row)
        {
            for (int col = 1; col < width - 1; ++col)
            {
                int index = row * width + col;
                ColorValue filtered_value = bilateral_filter(
                    input_jpeg.pixels, row, col, width, channel);
                output_pixels[index].set_channel(channel, filtered_value);
            }
        }
    }
    // for (int row = 1; row < height - 1; ++row)
    // {
    //     for (int col = 1; col < width - 1; ++col)
    //     {
    //         int index = row * width + col;
    //         for (int channel = 0; channel < num_channels; ++channel)
    //         {
    //             ColorValue filtered_value = bilateral_filter(
    //                 input_jpeg.pixels, row, col, width, channel);
    //             output_pixels[index].set_channel(channel, filtered_value);
    //         }
    //     }
    // }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JpegAOS output_jpeg{output_pixels, width, height, num_channels,
                        input_jpeg.color_space};
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Cleanup
    delete[] output_pixels;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
