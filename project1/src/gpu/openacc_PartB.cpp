//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// OpenACC implementation of image filtering on JPEG
//

#include <memory.h>
#include <cstring>
#include <chrono>
#include <cmath>
#include <iostream>
#include <openacc.h>

#include "../utils.hpp"

#pragma acc routine seq
float acc_linear_filter(unsigned char* image_buffer,
                        const float (&filter)[FILTERSIZE][FILTERSIZE],
                        int pixel_id, int width, int num_channels)
{
    float sum = 0;
    int line_width = width * num_channels;
    sum += image_buffer[pixel_id] * filter[1][1];
    sum += image_buffer[pixel_id - num_channels] * filter[1][0];
    sum += image_buffer[pixel_id + num_channels] * filter[1][2];
    sum += image_buffer[pixel_id - line_width] * filter[0][1];
    sum += image_buffer[pixel_id - line_width - num_channels] * filter[0][0];
    sum += image_buffer[pixel_id - line_width + num_channels] * filter[0][2];
    sum += image_buffer[pixel_id + line_width] * filter[2][1];
    sum += image_buffer[pixel_id + line_width - num_channels] * filter[2][0];
    sum += image_buffer[pixel_id + line_width + num_channels] * filter[2][2];
    return sum;
}

#pragma acc routine seq
unsigned char acc_clamp_pixel_value(float pixel)
{
    return pixel > 255 ? 255
           : pixel < 0 ? 0
                       : static_cast<unsigned char>(pixel);
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

    // Apply the filter to the image
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    size_t buffer_size = width * height * num_channels;
    unsigned char* filteredImage = new unsigned char[buffer_size];
    unsigned char* buffer = new unsigned char[buffer_size];

    memset(filteredImage, 0, buffer_size);
    memcpy(buffer, input_jpeg.buffer, buffer_size);
    delete[] input_jpeg.buffer;

#pragma acc enter data copyin(filteredImage[0 : buffer_size], \
                              buffer[0 : buffer_size],        \
                              filter[0 : FILTERSIZE][0 : FILTERSIZE])

#pragma acc update device(filteredImage[0 : buffer_size], \
                          buffer[0 : buffer_size],        \
                          filter[0 : FILTERSIZE][0 : FILTERSIZE])
    auto start_time = std::chrono::high_resolution_clock::now();
#pragma acc parallel present(                                \
    filteredImage[0 : buffer_size], buffer[0 : buffer_size], \
    filter[0 : FILTERSIZE][0 : FILTERSIZE]) num_gangs(1024)
    {
#pragma acc loop independent
        for (int y = 1; y < height - 1; y++)
        {
#pragma acc loop independent
            for (int x = 1; x < width - 1; x++)
            {
                int r_id = (y * width + x) * num_channels;
                int g_id = r_id + 1;
                int b_id = r_id + 2;

                float r_sum = acc_linear_filter(buffer, filter, r_id, width,
                                                num_channels);

                float g_sum = acc_linear_filter(buffer, filter, g_id, width,
                                                num_channels);

                float b_sum = acc_linear_filter(buffer, filter, b_id, width,
                                                num_channels);

                filteredImage[r_id] = acc_clamp_pixel_value(r_sum);
                filteredImage[g_id] = acc_clamp_pixel_value(g_sum);
                filteredImage[b_id] = acc_clamp_pixel_value(b_sum);
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
#pragma acc update self(filteredImage[0 : buffer_size])

#pragma acc exit data copyout(filteredImage[0 : buffer_size])

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
    delete[] buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
