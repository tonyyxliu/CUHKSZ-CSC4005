#ifndef OPS_HPP
#define OPS_HPP

#include "utils.hpp"

/**
 * @brief GEMM (General Matrix Multiply) operation.
 * 
 * @param A 1D input array of size (batch_num * (m * n)).
 * @param B 1D input array of size ((m * n) * k).
 * @param Out 1D output array of size (batch_num * k).
 * @param batch Number of batch.
 * @param mn Number of (m * n).
 * @param k Number of k.
 */
void gemm(const float* A, const float* B, float* Out, size_t batch, size_t mn, size_t k);

/**
 * @brief Add_bias operation.
 * 
 * @param A 1D input array of size (batch_num * out_dim).
 * @param B 1D output array of size (batch_num * out_dim).
 * @param bias 1D input array of size out_dim.
 * @param batch Number of batch.
 * @param out_dim Number of output dimension.
 */
void add_bias(float* A, float* B, const float* bias, size_t batch, size_t out_dim);

/**
 * @brief Relu operation.
 * 
 * @param A 1D input array of size (batch_num * out_dim).
 * @param B 1D output array of size (batch_num * out_dim).
 * @param size Size of A.
 */
void Relu(float* A, float* B, size_t size);

/**
 * @brief Softmax operation.
 * 
 * @param A 1D input array of size (batch_num * out_dim).
 * @param B 1D output array of size (batch_num * out_dim).
 * @param batch Number of batch.
 * @param out_dim Number of output dimension.
 */
void Softmax(float* A, float* B, size_t batch, size_t out_dim);

/**
 * @brief Vector to one hot matrix operation.
 * 
 * @param A 1D input array of size (batch_num).
 * @param B 1D output array of size (batch_num * out_dim).
 * @param batch Number of batch.
 * @param out_dim Number of output dimension.
 */
void vector_to_one_hot_matrix(const unsigned char* A, float* B, size_t batch, size_t out_dim);

/**
 * @brief Cross entropy loss operation.
 * 
 * @param A 1D input array of size (batch_num * out_dim).
 * @param B 1D input array of size (batch_num * out_dim).
 * @param Loss 1D output array of size (batch_num).
 * @param batch Number of batch.
 * @param out_dim Number of output dimension.
 */
void cross_entropy_loss(const float* A, const float* B, float* Loss, size_t batch, size_t out_dim);

/**
 * @brief Cross entropy loss gradient operation.
 * 
 * @param A 1D input array of size (batch_num * out_dim). Predicted value.
 * @param B 1D input array of size (batch_num * out_dim). True value.
 * @param Grad 1D output array of size (batch_num * out_dim).
 * @param batch Number of batch.
 * @param out_dim Number of output dimension.
 */
void cross_entropy_loss_grad(const float* A, const float* B, float* Grad, size_t batch, size_t out_dim);

/**
 * @brief Update bias operation.
 * 
 * @param Bias 1D input array of size out_dim.
 * @param Output_Grad 1D input array of size (batch_num * out_dim).
 * @param batch Number of batch.
 * @param lr Learning rate.
 * @param out_dim Number of output dimension.
 */
void update_bias(float* Bias, const float* Output_Grad, size_t batch, float lr, size_t out_dim);

/**
 * @brief Get Input Gradient from Output Grad.
 *          Hint: You will need the weight matrix to calculate the input gradient from the output gradient.(Backpropagation Stage)
 * 
 * @param Weight 1D input array of size (in_dim * out_dim).
 * @param Output_Grad 1D input array of size (batch_num * out_dim).
 * @param Input 1D input array of size (batch_num * in_dim).
 * @param Input_Grad 1D output array of size (batch_num * in_dim).
 * @param batch Number of batch.
 * @param in_dim Number of input dimension.
 * @param out_dim Number of output dimension.
 */
void input_grad(const float* Weight, const float* Output_Grad, float* Input, float* Input_Grad, size_t batch, size_t in_dim, size_t out_dim);

/**
 * @brief Update weight operation.
 * 
 * @param Weight 1D input array of size (in_dim * out_dim).
 * @param Output_Grad 1D input array of size (batch_num * out_dim).
 * @param Input 1D input array of size (batch_num * in_dim).
 * @param batch Number of batch.
 * @param lr Learning rate.
 * @param in_dim Number of input dimension.
 * @param out_dim Number of output dimension.
 */
void update_weight(float* Weight, const float* Output_Grad, const float* Input, size_t batch, float lr, size_t in_dim, size_t out_dim);

/**
 * @brief Relu Gradient operation.
 *      Hint: This is an in-place operation.(Grad input is also output)
 * 
 * @param A 1D input array of size (batch_num * out_dim).
 * @param Grad 1D output array of size (batch_num * out_dim).
 * @param batch Number of batch.
 * @param out_dim Number of output dimension.
 */
void relu_grad(const float* A, float* Grad, size_t batch, size_t out_dim);

/**
 * @brief Compute mean accuracy operation.
 *
 * @param result 1D input array of size (images_num).
 * @param labels_array 1D input array of size (images_num).
 * @param images_num Number of images.
 * @param num_classes Number of classes.
 */
float mean_acc(const unsigned char* result, const unsigned char* labels_array, size_t images_num, size_t num_classes);

/**
 * @brief Argmax operation. You can see this function like one-hot matrix to vector.
 * 
 * @param A 1D input array of size (num_classes * images_num).
 * @param B 1D output array of size images_num.
 * @param num_classes Number of classes.
 * @param images_num Number of images.
 */
void argmax(const float* A, unsigned char* B, size_t num_classes, size_t images_num);

#endif // OPS_HPP
