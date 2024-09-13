//
// Created by Liu Yuxuan on 2023/9/15
// Updated by Liu Yuxuan on 2024/9/9
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// General utility functions related to jpeg parsing and image filtering
//

#ifndef CSC4005_PROJECT_1_UTILS_HPP
#define CSC4005_PROJECT_1_UTILS_HPP

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <jpeglib.h>

#define ColorValue unsigned char
#define FILTERSIZE 3
#define M_PI 3.14159265358979323846
const float filter[FILTERSIZE][FILTERSIZE] = { // [Linear Filter Only]
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}};
const float SIGMA_D = 1.7;  // [Bilateral Filter Only] spatial stddev
const float SIGMA_R = 50.0; // [Bilateral Filter Only] intensity stddev

/**
 * Single RGB Pixel
 *
 * used in JpegAOS to store as array of structure
 */
struct Pixel
{
    ColorValue r;
    ColorValue g;
    ColorValue b;

    /**
     * get the single channel value
     * @param channel:
     *  channel = 0 --> R
     *  channel = 1 --> G
     *  channel = 2 --> B
     * @return
     */
    ColorValue get_channel(int channel) const
    {
        switch (channel)
        {
            case 0: return r;
            case 1: return g;
            case 2: return b;
            default: {
                std::cerr << "invalid channel: " << channel << std::endl;
                return 0;
            }
        }
    }

    /**
     * set the value of a specific channel
     * @param channel:
     *  channel = 0 --> R
     *  channel = 1 --> G
     *  channel = 2 --> B
     * @param value: value to set for the specific channel
     */
    void set_channel(int channel, ColorValue value)
    {
        switch (channel)
        {
            case 0: r = value; return;
            case 1: g = value; return;
            case 2: b = value; return;
            default: {
                std::cerr << "invalid channel: " << channel << std::endl;
                return;
            }
        }
    }
};

/**
 * JPEG image info (One way of Array-of-Structure form)
 *
 * all r,g,b values are stored in one array
 */
struct JPEGMeta
{
    ColorValue* buffer; // size = width * height * num_channels
    int width;
    int height;
    int num_channels;
    J_COLOR_SPACE color_space;
};

/**
 * JPEG image info (Array-of-Structure form)
 *
 * RGB info of a single Pixel is stored as a Pixel struct, and buffer is an *
 * array of Pixel
 */
struct JpegAOS
{
    Pixel* pixels; // size = width * height
    int width;
    int height;
    int num_channels;
    J_COLOR_SPACE color_space;
};

/**
 * JPEG Image Info (Structure-of-Array form)
 */
struct JpegSOA
{
    ColorValue* r_values; // size = width * height
    ColorValue* g_values; // size = width * height
    ColorValue* b_values; // size = width * height
    int width;
    int height;
    int num_channels;
    J_COLOR_SPACE color_space;

    /**
     * Get the list of values of the specific channel
     * @param channel
     * channel = 0 --> R
     * channel = 1 --> G
     * channel = 2 --> B
     * @return values of the specific channel
     */
    ColorValue* get_channel(int channel) const
    {
        switch (channel)
        {
            case 0: return r_values;
            case 1: return g_values;
            case 2: return b_values;
            default: {
                std::cerr << "invalid channel: " << channel << std::endl;
                return nullptr;
            }
        }
    }

    /**
     * get a value of specific channel, specific index
     * @param channel
     * channel = 0 --> R
     * channel = 1 --> G
     * channel = 2 --> B
     * @param index: element index in the array
     * @return
     */
    ColorValue get_value(int channel, int index) const
    {
        switch (channel)
        {
            case 0: return r_values[index];
            case 1: return g_values[index];
            case 2: return b_values[index];
            default: {
                std::cerr << "invalid channel: " << channel << std::endl;
                return 0;
            }
        }
    }

    /**
     * set value for a specific index in a specific channel
     * @param channel
     * channel = 0 --> R
     * channel = 1 --> G
     * channel = 2 --> B
     * @param index
     * @param value
     */
    void set_value(int channel, int index, ColorValue value)
    {
        switch (channel)
        {
            case 0: r_values[index] = value; return;
            case 1: g_values[index] = value; return;
            case 2: b_values[index] = value; return;
            default: {
                std::cerr << "invalid channel: " << channel << std::endl;
                return;
            }
        }
    }
};

JPEGMeta read_from_jpeg(const char* filepath);

JpegAOS read_jpeg_aos(const char* filepath);

JpegSOA read_jpeg_soa(const char* filepath);

int export_jpeg(const JPEGMeta& data, const char* filepath);

int export_jpeg(const JpegAOS& data, const char* filepath);

int export_jpeg(const JpegSOA& data, const char* filepath);

ColorValue bilateral_filter(const Pixel* pixels, int row, int col, int width,
                            int channel);

ColorValue bilateral_filter(const ColorValue* values, int row, int col,
                            int width);

float linear_filter(const ColorValue* values,
                    const float (&filter)[FILTERSIZE][FILTERSIZE], int pixel_id,
                    int width, int num_channels);

ColorValue linear_filter(const ColorValue* values,
                         const float (&filter)[FILTERSIZE][FILTERSIZE],
                         int index, int width);

ColorValue clamp_pixel_value(float Pixel);

#endif // CSC4005_PROJECT_1_UTILS_HPP
