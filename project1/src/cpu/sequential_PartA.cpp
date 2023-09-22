//
// Created by Liu Yuxuan on 2023/9/15.
// Email: yuxuanliu1@link.cuhk.edu.cm
//
// Sequential implementation of converting a JPEG picture from RGB to gray
//

#include <iostream>
#include <chrono>

#include "utils.hpp"

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read input JPEG image
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    // Computation: RGB to Gray
    auto grayImage = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        unsigned char r = input_jpeg.buffer[i * input_jpeg.num_channels];
        unsigned char g = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        unsigned char b = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
        grayImage[i] = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    // Write GrayImage to output JPEG
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{grayImage, input_jpeg.width, input_jpeg.height, 1, JCS_GRAYSCALE};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Release allocated memory
    delete[] input_jpeg.buffer;
    delete[] grayImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}

