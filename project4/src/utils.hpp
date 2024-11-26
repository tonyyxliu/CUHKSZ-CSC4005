#ifndef UTLIS
#define UTLIS

#include <memory.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <random>

/**
 * Contain the MNIST Data.
 * Member:
 *    images_matrix: 1D float array containing the loaded
 *        data.  The dimensionality of the data should be
 *        (num_examples x input_dim) where 'input_dim' is the full
 *        dimension of the data, e.g., since MNIST images are 28x28, it
 *        will be 784.  Values should be of type float, and the data
 *        should be normalized to have a minimum value of 0.0 and a
 *        maximum value of 1.0 (i.e., scale original values of 0 to 0.0
 *        and 255 to 1.0).
 *
 *    labels_array: 1D unsigned char array containing the
 *        labels of the examples.  Values should be of type uint8 and
 *        for MNIST will contain the values 0-9.
 */
class DataSet
{
public:
    float *images_matrix;
    unsigned char *labels_array;
    size_t images_num;
    size_t input_dim;

    DataSet(size_t images_num, size_t input_dim);
    ~DataSet();
};

uint32_t swap_endian(uint32_t val);

DataSet *parse_mnist(const std::string &image_filename, const std::string &label_filename);

void print_matrix(const float *A, size_t m, size_t n);

#endif
