//
// Created by Liu Yuxuan on 2023/9/15
// Modified by Liu Yuxuan on 2024/9/12
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Here are some general utility functions for jpeg parsing and image filtering
//

#include "utils.hpp"

/**
 * Read buffer data and other metadata from JPEG file
 * @param filepath
 * @return
 */
JPEGMeta read_from_jpeg(const char* filepath)
{
    // Open file to read from
    FILE* file = fopen(filepath, "rb");
    if (file == nullptr)
    {
        std::cout << "Failed to open jpeg image: " << filepath << std::endl;
        return {nullptr, 0, 0, 0};
    }
    // Initialize JPEG Decoder
    struct jpeg_decompress_struct cinfo
    {
    };
    struct jpeg_error_mgr jerr
    {
    };
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, file);
    // Read JPEG Header
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);
    int width = cinfo.output_width;
    int height = cinfo.output_height;
    int num_channels = cinfo.output_components;
    // Read RGB buffer data from JPEG
    auto rgbImage = new unsigned char[width * height * num_channels];
    while (cinfo.output_scanline < cinfo.output_height)
    {
        unsigned char* rowPtr =
            rgbImage + cinfo.output_scanline * width * num_channels;
        jpeg_read_scanlines(&cinfo, &rowPtr, 1);
    }
    fclose(file); // Close jpeg file
    return {rgbImage, width, height, num_channels, cinfo.out_color_space};
}

/**
 * Read JPEG file and output its array-of-structure form
 * @param filepath
 * @return
 */
JpegAOS read_jpeg_aos(const char* filepath)
{
    // Open file to read from
    FILE* file = fopen(filepath, "rb");
    if (file == nullptr)
    {
        std::cerr << "Failed to open jpeg image: " << filepath << std::endl;
        return {nullptr, 0, 0, 0};
    }
    // Initialize JPEG Decoder & Read JPEG Header
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, file);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);
    int width = cinfo.output_width;
    int height = cinfo.output_height;
    int num_channels = cinfo.output_components;
    if (num_channels > 3)
    {
        std::cerr << "Do not support images with more than 3 channels"
                  << std::endl;
        return {nullptr, 0, 0, 0};
    }
    // Read RGB buffer data from JPEG line by line
    auto buffer = new ColorValue[width * num_channels]; // single line buffer
    Pixel* pixels = new Pixel[width * height];
    int row = 0;
    while (cinfo.output_scanline < height)
    {
        jpeg_read_scanlines(&cinfo, &buffer, 1);
        for (int col = 0; col < width; ++col)
        {
            int index = row * width + col;
            for (int channel = 0; channel < num_channels; ++channel)
            {
                pixels[index].set_channel(channel,
                                          buffer[col * num_channels + channel]);
            }
        }
        ++row;
    }
    // Cleanup & Close jpeg file
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(file);
    delete[] buffer;
    return {pixels, width, height, num_channels, cinfo.out_color_space};
}

/**
 * Read JPEG file and output its structure-of-array form
 * @param filepath
 * @return
 */
JpegSOA read_jpeg_soa(const char* filepath)
{
    // Open file to read from
    FILE* file = fopen(filepath, "rb");
    if (file == nullptr)
    {
        std::cout << "Failed to open jpeg image: " << filepath << std::endl;
        return {nullptr, nullptr, nullptr, 0, 0, 0};
    }
    // Initialize JPEG Decoder & Read JPEG Header
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, file);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);
    int width = cinfo.output_width;
    int height = cinfo.output_height;
    int num_channels = cinfo.output_components;
    // Read RGB buffer data from JPEG line by line
    auto buffer = new ColorValue[width * num_channels]; // single line buffer
    ColorValue* r_values = new ColorValue[width * height];
    ColorValue* g_values = new ColorValue[width * height];
    ColorValue* b_values = new ColorValue[width * height];
    JpegSOA jpeg{r_values,
                 g_values,
                 b_values,
                 width,
                 height,
                 num_channels,
                 cinfo.out_color_space};
    int row = 0;
    while (cinfo.output_scanline < height)
    {
        jpeg_read_scanlines(&cinfo, &buffer, 1);
        for (int col = 0; col < width; ++col)
        {
            int index = row * width + col;
            for (int channel = 0; channel < num_channels; ++channel)
            {
                jpeg.set_value(channel, index,
                               buffer[col * num_channels + channel]);
            }
        }
        ++row;
    }
    // Cleanup & Close jpeg file
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(file);
    delete[] buffer;
    return jpeg;
}

/**
 * Write buffer data into JPEG file stored under the filepath
 * @param data
 * @param filepath
 * @return 0 on success, -1 on error
 */
int export_jpeg(const JPEGMeta& data, const char* filepath)
{
    // Open jpeg file to write to
    FILE* outputFile = fopen(filepath, "wb");
    if (outputFile == nullptr)
    {
        std::cout << "Failed to create output jpeg image: " << filepath
                  << std::endl;
        return -1;
    }
    // Initialize JPEG Header
    struct jpeg_compress_struct cinfoOut;
    struct jpeg_error_mgr jerrOut;
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
    while (cinfoOut.next_scanline < data.height)
    {
        unsigned char* rowPtr = data.buffer + cinfoOut.next_scanline *
                                                  data.width *
                                                  data.num_channels;
        jpeg_write_scanlines(&cinfoOut, &rowPtr, 1);
    }
    jpeg_finish_compress(&cinfoOut);
    jpeg_destroy_compress(&cinfoOut);
    fclose(outputFile); // Close jpeg file
    return 0;
}

/**
 * Write buffer data into JPEG file stored under the filepath
 * @param data
 * @param filepath
 * @return 0 on success, -1 on error
 */
int export_jpeg(const JpegAOS& data, const char* filepath)
{
    // Open jpeg file to write to
    FILE* outputFile = fopen(filepath, "wb");
    if (outputFile == nullptr)
    {
        std::cout << "Failed to create output jpeg image: " << filepath
                  << std::endl;
        return -1;
    }
    // Initialize JPEG Header
    struct jpeg_compress_struct cinfoOut;
    struct jpeg_error_mgr jerrOut;
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
    auto buffer =
        new ColorValue[data.width * data.num_channels]; // single line buffer
    int row = 0;
    while (cinfoOut.next_scanline < data.height)
    {
        for (int col = 0; col < data.width; ++col)
        {
            Pixel pixel = data.pixels[row * data.width + col];
            for (int channel = 0; channel < data.num_channels; ++channel)
                buffer[col * data.num_channels + channel] =
                    pixel.get_channel(channel);
        }
        jpeg_write_scanlines(&cinfoOut, &buffer, 1);
        ++row;
    }
    // Cleanup
    jpeg_finish_compress(&cinfoOut);
    jpeg_destroy_compress(&cinfoOut);
    fclose(outputFile); // Close jpeg file
    delete[] buffer;
    return 0;
}

/**
 * Write buffer data into JPEG file stored under the filepath
 * @param data
 * @param filepath
 * @return 0 on success, -1 on error
 */
int export_jpeg(const JpegSOA& data, const char* filepath)
{
    // Open jpeg file to write to
    FILE* outputFile = fopen(filepath, "wb");
    if (outputFile == nullptr)
    {
        std::cout << "Failed to create output jpeg image: " << filepath
                  << std::endl;
        return -1;
    }
    // Initialize JPEG Header
    struct jpeg_compress_struct cinfoOut;
    struct jpeg_error_mgr jerrOut;
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
    auto buffer =
        new ColorValue[data.width * data.num_channels]; // single line buffer
    int row = 0;
    while (cinfoOut.next_scanline < data.height)
    {
        for (int col = 0; col < data.width; ++col)
        {
            int index = row * data.width + col;
            for (int channel = 0; channel < data.num_channels; ++channel)
                buffer[col * data.num_channels + channel] =
                    data.get_value(channel, index);
        }
        jpeg_write_scanlines(&cinfoOut, &buffer, 1);
        ++row;
    }
    // Cleanup
    jpeg_finish_compress(&cinfoOut);
    jpeg_destroy_compress(&cinfoOut);
    fclose(outputFile); // Close jpeg file
    delete[] buffer;
    return 0;
}

/**
 * Perform bilateral filter on a single Pixel (Array-of-Structure form)
 *
 * channel = 0 --> R
 * channel = 1 --> G
 * channel = 2 --> B
 *
 * @return filtered pixel value
 */
ColorValue bilateral_filter(const Pixel* pixels, int row, int col, int width,
                            int channel)
{
    ColorValue value_11 =
        pixels[(row - 1) * width + (col - 1)].get_channel(channel);
    ColorValue value_12 = pixels[(row - 1) * width + col].get_channel(channel);
    ColorValue value_13 =
        pixels[(row - 1) * width + (col + 1)].get_channel(channel);
    ColorValue value_21 = pixels[row * width + (col - 1)].get_channel(channel);
    ColorValue value_22 = pixels[row * width + col].get_channel(channel);
    ColorValue value_23 = pixels[row * width + (col + 1)].get_channel(channel);
    ColorValue value_31 =
        pixels[(row + 1) * width + (col - 1)].get_channel(channel);
    ColorValue value_32 = pixels[(row + 1) * width + col].get_channel(channel);
    ColorValue value_33 =
        pixels[(row + 1) * width + (col + 1)].get_channel(channel);
    // Spatial Weights
    float w_spatial_border = expf(-0.5 / powf(SIGMA_D, 2));
    float w_spatial_corner = expf(-1.0 / powf(SIGMA_D, 2));
    // Intensity Weights
    ColorValue center_value = value_22;
    float w_11 = w_spatial_corner * expf(powf(center_value - value_11, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_12 = w_spatial_border * expf(powf(center_value - value_12, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_13 = w_spatial_corner * expf(powf(center_value - value_13, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_21 = w_spatial_border * expf(powf(center_value - value_21, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_22 = 1.0;
    float w_23 = w_spatial_border * expf(powf(center_value - value_23, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_31 = w_spatial_corner * expf(powf(center_value - value_31, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_32 = w_spatial_border * expf(powf(center_value - value_32, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_33 = w_spatial_corner * expf(powf(center_value - value_33, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float sum_weights =
        w_11 + w_12 + w_13 + w_21 + w_22 + w_23 + w_31 + w_32 + w_33;
    // Calculate filtered value
    float filtered_value =
        (w_11 * value_11 + w_12 * value_12 + w_13 * value_13 + w_21 * value_21 +
         w_22 * center_value + w_23 * value_23 + w_31 * value_31 +
         w_32 * value_32 + w_33 * value_33) /
        sum_weights;
    return clamp_pixel_value(filtered_value);
}

/**
 * Perform bilateral filter on a single Pixel (Structure-of-Array form)
 *
 * @return filtered pixel value
 */
ColorValue bilateral_filter(const ColorValue* values, int row, int col,
                            int width)
{
    ColorValue value_11 = values[(row - 1) * width + (col - 1)];
    ColorValue value_12 = values[(row - 1) * width + col];
    ColorValue value_13 = values[(row - 1) * width + (col + 1)];
    ColorValue value_21 = values[row * width + (col - 1)];
    ColorValue value_22 = values[row * width + col];
    ColorValue value_23 = values[row * width + (col + 1)];
    ColorValue value_31 = values[(row + 1) * width + (col - 1)];
    ColorValue value_32 = values[(row + 1) * width + col];
    ColorValue value_33 = values[(row + 1) * width + (col + 1)];
    // Spatial Weights
    float w_spatial_border = expf(-0.5 / powf(SIGMA_D, 2));
    float w_spatial_corner = expf(-1.0 / powf(SIGMA_D, 2));
    // Intensity Weights
    ColorValue center_value = value_22;
    float w_11 = w_spatial_corner * expf(powf(center_value - value_11, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_12 = w_spatial_border * expf(powf(center_value - value_12, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_13 = w_spatial_corner * expf(powf(center_value - value_13, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_21 = w_spatial_border * expf(powf(center_value - value_21, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_22 = 1.0;
    float w_23 = w_spatial_border * expf(powf(center_value - value_23, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_31 = w_spatial_corner * expf(powf(center_value - value_31, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_32 = w_spatial_border * expf(powf(center_value - value_32, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_33 = w_spatial_corner * expf(powf(center_value - value_33, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float sum_weights =
        w_11 + w_12 + w_13 + w_21 + w_22 + w_23 + w_31 + w_32 + w_33;
    // Calculate filtered value
    float filtered_value =
        (w_11 * value_11 + w_12 * value_12 + w_13 * value_13 + w_21 * value_21 +
         w_22 * center_value + w_23 * value_23 + w_31 * value_31 +
         w_32 * value_32 + w_33 * value_33) /
        sum_weights;
    return clamp_pixel_value(filtered_value);
}

float linear_filter(const ColorValue* values,
                    const float (&filter)[FILTERSIZE][FILTERSIZE], int pixel_id,
                    int width, int num_channels)
{
    float sum = 0;
    int line_width = width * num_channels;
    sum += values[pixel_id] * filter[1][1];
    sum += values[pixel_id - num_channels] * filter[1][0];
    sum += values[pixel_id + num_channels] * filter[1][2];
    sum += values[pixel_id - line_width] * filter[0][1];
    sum += values[pixel_id - line_width - num_channels] * filter[0][0];
    sum += values[pixel_id - line_width + num_channels] * filter[0][2];
    sum += values[pixel_id + line_width] * filter[2][1];
    sum += values[pixel_id + line_width - num_channels] * filter[2][0];
    sum += values[pixel_id + line_width + num_channels] * filter[2][2];
    return sum;
}

ColorValue linear_filter(const ColorValue* values,
                         const float (&filter)[FILTERSIZE][FILTERSIZE],
                         int index, int width)
{
    float sum = 0;
    sum += values[index - width - 1] * filter[0][0];
    sum += values[index - width] * filter[0][1];
    sum += values[index - width + 1] * filter[0][2];
    sum += values[index - 1] * filter[1][0];
    sum += values[index] * filter[1][1];
    sum += values[index + 1] * filter[1][2];
    sum += values[index + width - 1] * filter[2][0];
    sum += values[index + width] * filter[2][1];
    sum += values[index + width + 1] * filter[2][2];
    return clamp_pixel_value(sum);
}

/**
 * Limit the range of input fp value within 0-255
 * @param value
 * @return integer value within 0-255
 */
ColorValue clamp_pixel_value(float value)
{
    return value > 255 ? 255 : value < 0 ? 0 : static_cast<ColorValue>(value);
}
