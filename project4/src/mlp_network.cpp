#include "mlp_network.hpp"

float mean_acc(const unsigned char* result, const unsigned char* labels_array, size_t images_num, size_t num_classes)
{
    float acc = 0;
    for (size_t i = 0; i < images_num; i++)
    {
        if (result[i] == labels_array[i])
        {
            acc += 1;
        }
    }
    return acc / images_num;
}

void argmax(const float* A, unsigned char* B, size_t num_classes, size_t images_num)
{
    size_t start_idx;
    float max_element;
    int max_index;
    for (size_t i = 0; i < images_num; ++i)
    {
        start_idx = i * num_classes;
        max_element = A[start_idx];
        max_index = 0;

        for (size_t j = 1; j < num_classes; ++j)
        {
            if (A[start_idx + j] > max_element)
            {
                max_element = A[start_idx + j];
                max_index = j;
            }
        }
        
        B[i] = static_cast<unsigned char>(max_index);
    }
}