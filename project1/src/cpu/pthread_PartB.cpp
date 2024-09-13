//
// Created by Zhang Na on 2023/9/15.
// Email: nazhang@link.cuhk.edu.cn
//
// Pthread implementation of smooth image filtering of JPEG
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <pthread.h>

#include "../utils.hpp"

struct ThreadData
{
    unsigned char* inputBuffer;
    unsigned char* outputBuffer;
    int width;
    int height;
    int num_channels;
    int start_row;
    int end_row;
};

void* grayscale_filter_thread_function(void* arg)
{
    ThreadData* data = (ThreadData*)arg;

    for (int y = data->start_row; y < data->end_row; y++)
    {
        for (int x = 0; x < data->width; x++)
        {
            int r_id = (y * data->width + x) * data->num_channels;
            int g_id = r_id + 1;
            int b_id = r_id + 2;

            float r_sum = linear_filter(data->inputBuffer, filter, r_id,
                                        data->width, data->num_channels);
            float g_sum = linear_filter(data->inputBuffer, filter, g_id,
                                        data->width, data->num_channels);
            float b_sum = linear_filter(data->inputBuffer, filter, b_id,
                                        data->width, data->num_channels);

            data->outputBuffer[r_id] = clamp_pixel_value(r_sum);
            data->outputBuffer[g_id] = clamp_pixel_value(g_sum);
            data->outputBuffer[b_id] = clamp_pixel_value(b_sum);
        }
    }
    return nullptr;
}

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

    int NUM_THREADS = std::stoi(argv[3]); // Convert the input to integer

    unsigned char* filteredImage =
        new unsigned char[input_jpeg.width * input_jpeg.height *
                          input_jpeg.num_channels];

    pthread_t* threads = new pthread_t[NUM_THREADS];
    ThreadData* threadData = new ThreadData[NUM_THREADS];
    int rowsPerThread = input_jpeg.height / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; i++)
    {
        threadData[i] = {input_jpeg.buffer,
                         filteredImage,
                         input_jpeg.width,
                         input_jpeg.height,
                         input_jpeg.num_channels,
                         i * rowsPerThread,
                         (i == NUM_THREADS - 1) ? input_jpeg.height
                                                : (i + 1) * rowsPerThread};
    }

    auto start_time =
        std::chrono::high_resolution_clock::now(); // Start time recording

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_create(&threads[i], NULL, grayscale_filter_thread_function,
                       &threadData[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    auto end_time =
        std::chrono::high_resolution_clock::now(); // End time recording

    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height,
                         input_jpeg.num_channels, input_jpeg.color_space};
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    delete[] threads;
    delete[] threadData;

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";

    return 0;
}
