//
// Created by Yang Yufan on 2023/9/16.
// Email: yufanyang1@link.cuhk.edu.cm
//
// SIMD (AVX2) implementation of transferring a JPEG picture from RGB to gray
//

#include <iostream>
#include <chrono>

#include <immintrin.h>

#include "utils.hpp"

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
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

    // Transform the RGB Contents to the gray contents
    auto grayImage = new unsigned char[input_jpeg.width * input_jpeg.height + 8];

    // Prepross, store reds, greens and blues separately
    auto reds = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto greens = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto blues = new unsigned char[input_jpeg.width * input_jpeg.height + 16];

    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        reds[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
        greens[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        blues[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    }

    // Set SIMD scalars, we use AVX2 instructions
    __m256 redScalar = _mm256_set1_ps(0.299f);
    __m256 greenScalar = _mm256_set1_ps(0.587f);
    __m256 blueScalar = _mm256_set1_ps(0.114f);

    // Mask used for shuffling when store int32s to u_int8 arrays
    // |0|0|0|4|0|0|0|3|0|0|0|2|0|0|0|1| -> |4|3|2|1|
    __m128i shuffle = _mm_setr_epi8(0, 4, 8, 12, 
                                    -1, -1, -1, -1, 
                                    -1, -1, -1, -1, 
                                    -1, -1, -1, -1);

    // Using SIMD to accelerate the transformation
    auto start_time = std::chrono::high_resolution_clock::now();    // Start recording time
    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i+=8) {
        // Load the 8 red chars to a 256 bits float register
        __m128i red_chars = _mm_loadu_si128((__m128i*) (reds+i));
        __m256i red_ints = _mm256_cvtepu8_epi32(red_chars);
        __m256 red_floats = _mm256_cvtepi32_ps(red_ints);
        // Multiply the red floats to the red scalar
        __m256 red_results = _mm256_mul_ps(red_floats, redScalar);

        // Load the 8 green chars to a 256 bits float register
        __m128i green_chars = _mm_loadu_si128((__m128i*) (greens+i));
        __m256i green_ints = _mm256_cvtepu8_epi32(green_chars);
        __m256 green_floats = _mm256_cvtepi32_ps(green_ints);
        // Multiply the green floats to the green scalar
        __m256 green_results = _mm256_mul_ps(green_floats, greenScalar);

        // Load the 8 blue chars to a 256 bits float register
        __m128i blue_chars = _mm_loadu_si128((__m128i*) (blues+i));
        __m256i blue_ints = _mm256_cvtepu8_epi32(blue_chars);
        __m256 blue_floats = _mm256_cvtepi32_ps(blue_ints);
        // Multiply the blue floats to the blue scalar
        __m256 blue_results = _mm256_mul_ps(blue_floats, blueScalar);

        // Add red, green and blue results
        __m256 add_results = _mm256_add_ps(red_results, green_results);
        add_results = _mm256_add_ps(add_results, blue_results);
        // Convert the float32 results to int32
        __m256i add_results_ints =  _mm256_cvtps_epi32(add_results);

        // Seperate the 256bits result to 2 128bits result
        __m128i low = _mm256_castsi256_si128(add_results_ints);
        __m128i high = _mm256_extracti128_si256(add_results_ints, 1);

        // shuffling int32s to u_int8s
        // |0|0|0|4|0|0|0|3|0|0|0|2|0|0|0|1| -> |4|3|2|1|
        __m128i trans_low = _mm_shuffle_epi8(low, shuffle);
        __m128i trans_high = _mm_shuffle_epi8(high, shuffle);

        // Store the results back to gray image
        _mm_storeu_si128((__m128i*)(&grayImage[i]), trans_low);
        _mm_storeu_si128((__m128i*)(&grayImage[i+4]), trans_high);
    }

    auto end_time = std::chrono::high_resolution_clock::now();  // Stop recording time
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save output Gray JPEG Image
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

