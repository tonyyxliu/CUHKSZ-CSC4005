//
// Created by Liu Yuxuan on 2023/9/15.
// Email: yuxuanliu1@link.cuhk.edu.cm
//
// This is the general utility function to read / write jpeg image with libjpeg
//

#include "utils.hpp"

/**
 * Read buffer data and other metadata from JPEG file
 * @param filepath
 * @return
 */
JPEGMeta read_from_jpeg(const char* filepath) {
    // Open file to read from
    FILE* file = fopen(filepath, "rb");
    if (file == NULL) {
        std::cout << "Failed to open the JPEG image" << std::endl;
        return {NULL, 0, 0, 0};
    }
    // Initialize JPEG Decoder
    struct jpeg_decompress_struct cinfo {};
    struct jpeg_error_mgr jerr {};
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, file);
    // Read JPEG Header
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);
    int width = cinfo.output_width;
    int height = cinfo.output_height;
    int numChannels = cinfo.output_components;
    // Read RGB buffer data from JPEG
    auto rgbImage = new unsigned char[width * height * numChannels];
    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char* rowPtr =
            rgbImage + cinfo.output_scanline * width * numChannels;
        jpeg_read_scanlines(&cinfo, &rowPtr, 1);
    }
    fclose(file);  // Close jpeg file
    return {rgbImage, width, height, numChannels, cinfo.out_color_space};
}

/**
 * Write buffer data into JPEG file stored under the filepath
 * @param data
 * @param filepath
 * @return 0 on success, -1 on error
 */
int write_to_jpeg(const JPEGMeta& data, const char* filepath) {
    // Open jpeg file to write to
    FILE* outputFile = fopen(filepath, "wb");
    if (outputFile == NULL) {
        std::cout << "Failed to output JPEG image" << std::endl;
        return -1;
    }
    // Initialize JPEG Header
    struct jpeg_compress_struct cinfoOut {};
    struct jpeg_error_mgr jerrOut {};
    cinfoOut.err = jpeg_std_error(&jerrOut);
    jpeg_create_compress(&cinfoOut);
    jpeg_stdio_dest(&cinfoOut, outputFile);
    cinfoOut.image_width = data.width;
    cinfoOut.image_height = data.height;
    cinfoOut.input_components = data.num_channels;
    cinfoOut.in_color_space = data.color_space;
    jpeg_set_defaults(&cinfoOut);
    jpeg_set_quality(&cinfoOut, 100, TRUE);
    jpeg_start_compress(&cinfoOut, TRUE);
    // Write buffer data to jpeg
    while (cinfoOut.next_scanline < cinfoOut.image_height) {
        unsigned char* rowPtr = data.buffer + cinfoOut.next_scanline *
                                                  data.width *
                                                  data.num_channels;
        jpeg_write_scanlines(&cinfoOut, &rowPtr, 1);
    }
    jpeg_finish_compress(&cinfoOut);
    jpeg_destroy_compress(&cinfoOut);
    fclose(outputFile);  // Close jpeg file
    return 0;
}

/**
 * Perform image filtering on size-3 filter
 * @param image_buffer
 * @param filter
 * @param pixel_id
 * @param width
 * @param num_channels
 * @return filtered value
 */
float get_pixel_matrix_sum(unsigned char* image_buffer,
                           const float (&filter)[FILTERSIZE][FILTERSIZE],
                           int pixel_id, int width, int num_channels) {
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

/**
 * Threshold the pixel value within [0,255]
 * float to unsigned char BTW
 * @param pixel
 * @return
 */
unsigned char clamp_pixel_value(float pixel) {
    return pixel > 255 ? 255
           : pixel < 0 ? 0
                       : static_cast<unsigned char>(pixel);
}
