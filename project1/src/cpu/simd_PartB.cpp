//
// Created by Yang Yufan on 2023/9/16.
// Email: yufanyang1@link.cuhk.edu.cm
//
// SIMD (AVX2) implementation of transferring a JPEG picture from RGB to gray
//

#include <immintrin.h>
#include <memory.h>

#include <chrono>
#include <iostream>

#include "../utils.hpp"

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    // Read JPEG File
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    // Filter the first division of the contents
    auto filteredImage =
        new unsigned char[input_jpeg.width * input_jpeg.height *
                          input_jpeg.num_channels];
    memset(filteredImage, 0,
           input_jpeg.width * input_jpeg.height * input_jpeg.num_channels);

    __m256 filter_register1 =
        _mm256_set_ps(filter[0][0], filter[0][0], filter[0][0], filter[0][1],
                      filter[0][1], filter[0][1], filter[0][2], filter[0][2]);

    __m256 filter_register2 =
        _mm256_set_ps(filter[1][0], filter[1][0], filter[1][0], filter[1][1],
                      filter[1][1], filter[1][1], filter[1][2], filter[1][2]);

    __m256 filter_register3 =
        _mm256_set_ps(filter[2][0], filter[2][0], filter[2][0], filter[2][1],
                      filter[2][1], filter[2][1], filter[2][2], filter[2][2]);

    __m128 filter_register4 =
        _mm_set_ps(filter[0][2], filter[1][2], filter[2][2], 0);

    float added_array[8];
    float left_array[4];

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int y = 1; y < input_jpeg.height - 1; y++) {
        for (int x = 1; x < input_jpeg.width - 1; x++) {
            int r_id = (y * input_jpeg.width + x) * input_jpeg.num_channels;
            int g_id = r_id + 1;
            int b_id = r_id + 2;

            float r_sum = 0, g_sum = 0, b_sum = 0;

            int line_width = input_jpeg.width * input_jpeg.num_channels;
            unsigned char* low_pos =
                &input_jpeg.buffer[r_id - line_width - input_jpeg.num_channels];
            unsigned char* mid_pos =
                &input_jpeg.buffer[r_id - input_jpeg.num_channels];
            unsigned char* high_pos =
                &input_jpeg.buffer[r_id + line_width - input_jpeg.num_channels];

            __m128i low_chars = _mm_loadu_si128((__m128i*)low_pos);
            __m256i low_ints = _mm256_cvtepu8_epi32(low_chars);
            __m256 low_floats = _mm256_cvtepi32_ps(low_ints);
            __m256 low_results = _mm256_mul_ps(low_floats, filter_register1);

            __m128i mid_chars = _mm_loadu_si128((__m128i*)mid_pos);
            __m256i mid_ints = _mm256_cvtepu8_epi32(mid_chars);
            __m256 mid_floats = _mm256_cvtepi32_ps(mid_ints);
            __m256 mid_results = _mm256_mul_ps(mid_floats, filter_register2);

            __m128i high_chars = _mm_loadu_si128((__m128i*)high_pos);
            __m256i high_ints = _mm256_cvtepu8_epi32(high_chars);
            __m256 high_floats = _mm256_cvtepi32_ps(high_ints);
            __m256 high_results = _mm256_mul_ps(high_floats, filter_register3);

            __m256 add_results = _mm256_add_ps(low_results, mid_results);
            add_results = _mm256_add_ps(add_results, high_results);
            _mm256_storeu_ps(added_array, add_results);

            __m128i left_ints =
                _mm_set_epi32(input_jpeg.buffer[r_id - line_width + 5],
                              input_jpeg.buffer[r_id + 5],
                              input_jpeg.buffer[r_id + line_width + 5], 0);
            __m128 left_floats = _mm_cvtepi32_ps(left_ints);
            __m128 left_results = _mm_mul_ps(left_floats, filter_register4);
            _mm_storeu_ps(left_array, left_results);

            r_sum += (added_array[0] + added_array[3] + added_array[6]);
            g_sum += (added_array[1] + added_array[4] + added_array[7]);
            b_sum += (added_array[2] + added_array[5]);
            b_sum += (left_array[0] + left_array[1] + left_array[2]);

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
    if (export_jpeg(output_jpeg, output_filepath)) {
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
