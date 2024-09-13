//
// Created by Zhang Na on 2023/9/15.
// Email: nazhang@link.cuhk.edu.cn
//
// OpenMP implementation of transforming a JPEG image from RGB to gray
//

#include <iostream>
#include <chrono>
#include <omp.h> // OpenMP header
#include "../utils.hpp"

int main(int argc, char** argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read input JPEG image
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    // Separate R, G, B channels into three continuous arrays
    auto rChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto gChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto bChannel = new unsigned char[input_jpeg.width * input_jpeg.height];

    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++)
    {
        rChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
        gChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        bChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    }

    // Transforming the R, G, B channels to Gray in parallel
    auto grayImage = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto start_time = std::chrono::high_resolution_clock::now();

#pragma omp parallel for default(none) \
    shared(rChannel, gChannel, bChannel, grayImage, input_jpeg)
    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++)
    {
        grayImage[i] = static_cast<unsigned char>(
            0.299 * rChannel[i] + 0.587 * gChannel[i] + 0.114 * bChannel[i]);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Save output JPEG GrayScale image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{grayImage, input_jpeg.width, input_jpeg.height, 1,
                         JCS_GRAYSCALE};
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to save output JPEG image\n";
        return -1;
    }

    // Release the allocated memory
    delete[] input_jpeg.buffer;
    delete[] rChannel;
    delete[] gChannel;
    delete[] bChannel;
    delete[] grayImage;

    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
