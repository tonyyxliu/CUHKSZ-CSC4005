//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// OpenACC implementation of transforming a JPEG image from RGB to gray
//

#include <iostream>
#include <chrono>

#include "utils.hpp"
// #include <openacc.h> // OpenACC Header

int main(int argc, char **argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read from input JPEG
    const char *input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    JPEGMeta input_jpeg = read_from_jpeg(input_filepath);
    // Computation: RGB to Gray
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    unsigned char *grayImage = new unsigned char[width * height];
    unsigned char *buffer = new unsigned char[width * height * num_channels];
    for (int i = 0; i < width * height * num_channels; i++)
    {
        buffer[i] = input_jpeg.buffer[i];
    }
#pragma acc enter data copyin(grayImage[0 : width * height], \
                              buffer[0 : width * height * num_channels])

#pragma acc update device(grayImage[0 : width * height], \
                          buffer[0 : width * height * num_channels])

    auto start_time = std::chrono::high_resolution_clock::now();
#pragma acc parallel present(grayImage[0 : width * height],             \
                             buffer[0 : width * height * num_channels]) \
    num_gangs(1024)
    {
#pragma acc loop independent
        for (int i = 0; i < width * height; i++)
        {
            grayImage[i] = (0.299 * buffer[i * num_channels] +
                            0.587 * buffer[i * num_channels + 1] +
                            0.114 * buffer[i * num_channels + 2]);
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
#pragma acc update self(grayImage[0 : width * height], \
                        buffer[0 : width * height * num_channels])

#pragma acc exit data copyout(grayImage[0 : width * height])

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Write GrayImage to output JPEG
    const char *output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{grayImage, input_jpeg.width, input_jpeg.height, 1,
                         JCS_GRAYSCALE};
    if (write_to_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Release allocated memory
    delete[] input_jpeg.buffer;
    delete[] grayImage;
    delete[] buffer;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
