//
// Created by Zhang Na on 2023/9/15.
// Email: nazhang@link.cuhk.edu.cn
//
// OpenMP implementation of smooth image filtering on JPEG
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <omp.h>

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
    auto input_jpeg = read_from_jpeg(input_filename);

    int NUM_THREADS = std::stoi(argv[3]);
    omp_set_num_threads(NUM_THREADS);

    unsigned char* grayscaleImage =
        new unsigned char[input_jpeg.width * input_jpeg.height *
                          input_jpeg.num_channels];

    auto start_time = std::chrono::high_resolution_clock::now();

#pragma omp parallel for shared(input_jpeg, grayscaleImage)
    for (int y = 0; y < input_jpeg.height; y++)
    {
        for (int x = 0; x < input_jpeg.width; x++)
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

            grayscaleImage[r_id] = clamp_pixel_value(r_sum);
            grayscaleImage[g_id] = clamp_pixel_value(g_sum);
            grayscaleImage[b_id] = clamp_pixel_value(b_sum);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{grayscaleImage, input_jpeg.width, input_jpeg.height,
                         input_jpeg.num_channels, input_jpeg.color_space};
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    delete[] input_jpeg.buffer;
    delete[] grayscaleImage;

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";

    return 0;
}
