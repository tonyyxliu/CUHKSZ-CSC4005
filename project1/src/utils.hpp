//
// Created by Liu Yuxuan on 2023/9/15.
// Email: yuxuanliu1@link.cuhk.edu.cm
//
// General utility function to read / write JPEG image with libjpeg
//

#ifndef CSC4005_PROJECT_1_UTILS_HPP
#define CSC4005_PROJECT_1_UTILS_HPP

#include <iostream>
#include <cstdio>
#include <cstdlib>

#include <jpeglib.h>

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

JPEGMeta read_from_jpeg(const char* filepath);

int write_to_jpeg(const JPEGMeta &data, const char* filepath);


#endif // CSC4005_PROJECT_1_UTILS_HPP
