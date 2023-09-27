//
// Created by Liu Yuxuan on 2023/9/15.
// Email: yuxuanliu1@link.cuhk.edu.cm
//
// General utility function to read / write JPEG image with libjpeg
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
    if (file == NULL)
        return {NULL, 0, 0, 0};
    // Initialize JPEG Decoder
    struct jpeg_decompress_struct cinfo{};
    struct jpeg_error_mgr jerr{};
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
        unsigned char* rowPtr = rgbImage + cinfo.output_scanline * width * numChannels;
        jpeg_read_scanlines(&cinfo, &rowPtr, 1);
    }
    fclose(file);   // Close jpeg file
    return {rgbImage, width, height, numChannels, cinfo.out_color_space};
}

/**
 * Write buffer data into JPEG file stored under the filepath
 * @param data
 * @param filepath
 * @return 0 on success, -1 on error
 */
int write_to_jpeg(const JPEGMeta &data, const char* filepath) {
    // Open jpeg file to write to
    FILE* outputFile = fopen(filepath, "wb");
    if (outputFile == NULL)
        return -1;
    // Initialize JPEG Header
    struct jpeg_compress_struct cinfoOut{};
    struct jpeg_error_mgr jerrOut{};
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
        unsigned char* rowPtr = data.buffer + cinfoOut.next_scanline * data.width * data.num_channels;
        jpeg_write_scanlines(&cinfoOut, &rowPtr, 1);
    }
    jpeg_finish_compress(&cinfoOut);
    jpeg_destroy_compress(&cinfoOut);
    fclose(outputFile); // Close jpeg file
    return 0;
}

