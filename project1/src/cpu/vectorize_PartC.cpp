//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// Sequential implementation of converting a JPEG from RGB to gray
// (Strcture-of-Array)
//

#include <memory.h>

#include <chrono>
#include <cmath>
#include <iostream>

#include "../utils.hpp"

/**
 * Perform bilateral filter on a single Pixel (Structure-of-Array form)
 *
 * @return filtered pixel value
 */
inline ColorValue bilateral_filter_vectorize(
    const ColorValue* const __restrict__ values, const int row, const int col,
    const int width)
{
    const float w_border = expf(-0.5f / (SIGMA_D * SIGMA_D));
    const float w_corner = expf(-1.0f / (SIGMA_D * SIGMA_D));
    const float sigma_r_sq_inv = -0.5f / (SIGMA_R * SIGMA_R);

    /**
     * TODO: vectorize the complicated computation
     * you can add pragma for loop unrolling like the one in PartB
     */

    return clamp_pixel_value(0.0);
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
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    // Apply the filter to the image
    const int width = input_jpeg.width;
    const int height = input_jpeg.height;
    const int num_channels = input_jpeg.num_channels;
    auto output_r_values = new ColorValue[width * height];
    auto output_g_values = new ColorValue[width * height];
    auto output_b_values = new ColorValue[width * height];
    JpegSOA output_jpeg{
        output_r_values, output_g_values, output_b_values,       width,
        height,          num_channels,    input_jpeg.color_space};
    auto start_time = std::chrono::high_resolution_clock::now();
    ColorValue* __restrict__ buf_r = input_jpeg.get_channel(0);
    ColorValue* __restrict__ buf_g = input_jpeg.get_channel(1);
    ColorValue* __restrict__ buf_b = input_jpeg.get_channel(2);
    ColorValue* output_r = output_jpeg.r_values;
    ColorValue* output_g = output_jpeg.g_values;
    ColorValue* output_b = output_jpeg.b_values;
    /* Pixels in the boundary can be ignored in this assignment */
    for (int row = 1; row < height - 1; ++row)
    {
#pragma GCC ivdep
        for (int col = 1; col < width - 1; ++col)
        {
            /**
             * TODO: you can choose to change the main loop content here or
             * simply fill in the bilateral_filter_vectorize.
             */
            ColorValue filtered_value_r =
                bilateral_filter_vectorize(buf_r, row, col, width);
            ColorValue filtered_value_g =
                bilateral_filter_vectorize(buf_g, row, col, width);
            ColorValue filtered_value_b =
                bilateral_filter_vectorize(buf_b, row, col, width);
            int index = row * width + col;
            output_r[index] = filtered_value_r;
            output_g[index] = filtered_value_g;
            output_b[index] = filtered_value_b;
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Cleanup
    delete[] output_r_values;
    delete[] output_g_values;
    delete[] output_b_values;
    // print execution time
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
