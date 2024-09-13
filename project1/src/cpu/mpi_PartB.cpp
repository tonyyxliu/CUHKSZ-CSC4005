//
// Created by Yang Yufan on 2023/9/16.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI implementation of transforming a JPEG image from RGB to gray
//

#include <memory.h>
#include <mpi.h> // MPI Header

#include <chrono>
#include <iostream>
#include <vector>

#include "utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

void set_filtered_image(unsigned char* filtered_image, unsigned char* image,
                        int width, int num_chanels, int start_line,
                        int end_line, int offset);

int main(int argc, char** argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    // Read JPEG File
    const char* input_filepath = argv[1];
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    // Divide the task
    // For example, there are 11 lines and 3 tasks,
    // we try to divide to 4 4 3 instead of 3 3 5
    int total_line_num = input_jpeg.height - 2;
    int line_per_task = total_line_num / numtasks;
    int left_line_num = total_line_num % numtasks;

    std::vector<int> cuts(numtasks + 1, 1);
    int divided_left_line_num = 0;

    for (int i = 0; i < numtasks; i++)
    {
        if (divided_left_line_num < left_line_num)
        {
            cuts[i + 1] = cuts[i] + line_per_task + 1;
            divided_left_line_num++;
        }
        else
            cuts[i + 1] = cuts[i] + line_per_task;
    }

    // The tasks for the master executor
    // 1. Filter the first division of the contents
    // 2. Receive the filtered contents from slave executors
    // 3. Write the filtered contents to the JPEG File
    if (taskid == MASTER)
    {
        std::cout << "Input file from: " << input_filepath << "\n";
        auto filteredImage =
            new unsigned char[input_jpeg.width * input_jpeg.height *
                              input_jpeg.num_channels];
        memset(filteredImage, 0,
               input_jpeg.width * input_jpeg.height * input_jpeg.num_channels);

        auto start_time = std::chrono::high_resolution_clock::now();

        // // Filter the first division of the contents
        set_filtered_image(filteredImage, input_jpeg.buffer, input_jpeg.width,
                           input_jpeg.num_channels, cuts[taskid],
                           cuts[taskid + 1], 0);

        // Receive the transformed Gray contents from each slave executors
        for (int i = MASTER + 1; i < numtasks; i++)
        {
            int line_width = input_jpeg.width * input_jpeg.num_channels;
            unsigned char* start_pos = filteredImage + cuts[i] * line_width;
            int length = (cuts[i + 1] - cuts[i]) * line_width;
            MPI_Recv(start_pos, length, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD,
                     &status);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                  start_time);

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
        delete[] input_jpeg.buffer;
        delete[] filteredImage;
        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count()
                  << " milliseconds\n";
    }
    // The tasks for the slave executor
    // 1. Filter a division of image
    // 2. Send the Filterd contents back to the master executor
    else
    {
        // Intialize the filtered image
        int length = input_jpeg.width * (cuts[taskid + 1] - cuts[taskid]) *
                     input_jpeg.num_channels;
        int offset = input_jpeg.width * cuts[taskid] * input_jpeg.num_channels;

        auto filteredImage = new unsigned char[length];
        memset(filteredImage, 0, length);

        // Filter a coresponding division
        set_filtered_image(filteredImage, input_jpeg.buffer, input_jpeg.width,
                           input_jpeg.num_channels, cuts[taskid],
                           cuts[taskid + 1], offset);

        // Send the filtered image back to the master
        MPI_Send(filteredImage, length, MPI_CHAR, MASTER, TAG_GATHER,
                 MPI_COMM_WORLD);

        // Release the memory
        delete[] filteredImage;
    }

    MPI_Finalize();
    return 0;
}

void set_filtered_image(unsigned char* filtered_image, unsigned char* image,
                        int width, int num_chanels, int start_line,
                        int end_line, int offset)
{
    for (int y = start_line; y < end_line; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            int r_id = (y * width + x) * num_chanels;
            int g_id = r_id + 1;
            int b_id = r_id + 2;

            float r_sum =
                linear_filter(image, filter, r_id, width, num_chanels);

            float g_sum =
                linear_filter(image, filter, g_id, width, num_chanels);

            float b_sum =
                linear_filter(image, filter, b_id, width, num_chanels);

            filtered_image[r_id - offset] = clamp_pixel_value(r_sum);
            filtered_image[g_id - offset] = clamp_pixel_value(g_sum);
            filtered_image[b_id - offset] = clamp_pixel_value(b_sum);
        }
    }
}
