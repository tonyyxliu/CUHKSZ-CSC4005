//
// Created by Liu Yuxuan on 2023/9/15.
// Email: yuxuanliu1@link.cuhk.edu.cm
//
// This is the general utility function to read / write jpeg image with libjpeg
//

#ifndef CSC4005_PROJECT_1_UTILS_HPP
#define CSC4005_PROJECT_1_UTILS_HPP

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <jpeglib.h>

#define FILTERSIZE 3

/**
 * Buffer data and other important metadata sufficient to build JPEG picture
 */
struct JPEGMeta {
    unsigned char* buffer;  // buffer data
    int width;
    int height;
    int num_channels;
    J_COLOR_SPACE color_space;
};

const float filter[FILTERSIZE][FILTERSIZE] = {  // Smooth
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}};

JPEGMeta read_from_jpeg(const char* filepath);

int write_to_jpeg(const JPEGMeta& data, const char* filepath);

float get_pixel_matrix_sum(unsigned char* image_buffer,
                           const float (&filter)[FILTERSIZE][FILTERSIZE],
                           int pixel_id, int width, int num_channels);

unsigned char clamp_pixel_value(float pixel);

#endif  // CSC4005_PROJECT_1_UTILS_HPP
