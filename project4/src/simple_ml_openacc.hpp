#ifndef SIMPLE_ML_OPENACC
#define SIMPLE_ML_OPENACC

#include "simple_ml_ext.hpp"

void matrix_dot_openacc(const float *A, const float *B, float *C, size_t m, size_t n, size_t k);

void matrix_dot_trans_openacc(const float *A, const float *B, float *C, size_t n, size_t m, size_t k);

void matrix_trans_dot_openacc(const float *A, const float *B, float *C, size_t m, size_t n, size_t k);

void matrix_minus_openacc(float *A, const float *B, size_t m, size_t n);

void matrix_mul_scalar_openacc(float *C, float scalar, size_t m, size_t n);

void matrix_div_scalar_openacc(float *C, float scalar, size_t m, size_t n);

void matrix_softmax_normalize_openacc(float *C, size_t m, size_t n);

void vector_to_one_hot_matrix_openacc(const unsigned char *y, float *Y, size_t m, size_t k);

void softmax_regression_epoch_openacc(const float *X, const unsigned char *y,
                                      float *theta, size_t m, size_t n, size_t k,
                                      float lr, size_t batch);

void train_softmax_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t epochs = 10, float lr = 0.5, size_t batch = 100);

float mean_softmax_loss_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes);

float mean_err_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes);

void matrix_mul_openacc(float *A, const float *B, size_t size);

void nn_epoch_openacc(const float *X,
                      const unsigned char *y,
                      float *W1,
                      float *W2,
                      size_t m,
                      size_t n,
                      size_t l,
                      size_t k,
                      float lr,
                      size_t batch);

void train_nn_openacc(const DataSet *train_data,
                      const DataSet *test_data,
                      size_t num_classes,
                      size_t hidden_dim = 500,
                      size_t epochs = 10,
                      float lr = 0.5,
                      size_t batch = 100);

#endif
