//
// Created by Liu Yuxuan on 2023/9/15.
// Email: yuxuanliu1@link.cuhk.edu.cm
//
// Sequential implementation of converting a JPEG picture from RGB to gray
//

#include <iostream>
#include <chrono>

#include "../utils.hpp"

/* [RGB to GrayScale only] */
const float GRAYSCALE_COEF_R = 0.299f;
const float GRAYSCALE_COEF_G = 0.587f;
const float GRAYSCALE_COEF_B = 0.114f;
/**
 * Hint: can we replace floating-point computation with integer?
 * The division of 1024 (2^10) can be achieved by shifting operation
 * For example:
 *      0.299 * r = 306 * r / 1024 = (306 * r) >> 10
 * And, how do we deal with the rounding issue? Think about it.
 */
const int GRAYSCALE_COEF_R_INT = 306; // 0.299 * 1024 = 306
const int GRAYSCALE_COEF_G_INT = 601; // 0.587 * 1024 = 601
const int GRAYSCALE_COEF_B_INT = 117; // 0.114 * 1024 = 117

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
    // Allocate memory for the grayscale image
    const int num_pixels = input_jpeg.width * input_jpeg.height;
    const int num_channel = 3;
    auto grayImage = new ColorValue[num_pixels];
    auto start_time = std::chrono::high_resolution_clock::now();
#pragma omp simd
    for (int i = 0; i < num_pixels; ++i)
    {
        int base_idx = i * num_channel;
        ColorValue r = input_jpeg.buffer[base_idx];
        ColorValue g = input_jpeg.buffer[base_idx + 1];
        ColorValue b = input_jpeg.buffer[base_idx + 2];

        grayImage[i] = static_cast<ColorValue>(
            std::round(GRAYSCALE_COEF_R * r + GRAYSCALE_COEF_G * g +
                       GRAYSCALE_COEF_B * b));

        // /* Maybe the hint of replacing float computation with int can be
        // written in this way */ grayImage[i] = (GRAYSCALE_COEF_R_INT * r +
        // GRAYSCALE_COEF_G_INT * g +
        //                 GRAYSCALE_COEF_B_INT * b) >>
        //                10;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    // Write GrayImage to output JPEG
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{grayImage, input_jpeg.width, input_jpeg.height, 1,
                         JCS_GRAYSCALE};
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Release allocated memory
    delete[] input_jpeg.buffer;
    delete[] grayImage;
    // Print execution time
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
