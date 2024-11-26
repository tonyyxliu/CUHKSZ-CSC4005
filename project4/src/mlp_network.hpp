// network.hpp
#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "utils.hpp"
#include "ops.hpp"

/**
 * @brief Train a neural network.
 * 
 * This function initializes the weights and biases, and trains the neural network for a specified number of epochs.
 * 
 * @param train_data Pointer to the training dataset.
 * @param test_data Pointer to the test dataset.
 * @param num_classes Number of output classes.
 * @param hidden_dim Hidden layer size.
 * @param epochs Number of training epochs.
 * @param lr Learning rate.
 * @param batch Size of SGD minibatch.
 */
void train_nn(const DataSet* train_data, const DataSet* test_data, unsigned long epochs, unsigned long hidden_units, unsigned long batch_size, float learning_rate, unsigned long seed);

/**
 * @brief Perform one epoch of training for a neural network.
 * 
 * This function modifies the W1 and W2 matrices in place using stochastic gradient descent (SGD).
 * 
 * @param input_array 1D input array of size (input_num x input_dim).
 * @param label_array 1D class label array of size (input_num).
 * @param Weight1 1D array of first layer weights, of shape (input_dim x hidden_dim).
 * @param Weight2 1D array of second layer weights, of shape (hidden_dim x num_classes).
 * @param bias1 1D array of first layer bias, of shape (hidden_dim).
 * @param bias2 1D array of second layer bias, of shape (num_classes).
 * @param input_num Number of input data.
 * @param input_dim Input data size.
 * @param hidden_num Hidden layer size.
 * @param class_num Number of classes.
 * @param lr Learning rate for SGD.
 * @param batch_num Size of SGD minibatch.
 */
void nn_epoch_cpp(const float* input_array, const unsigned char* label_array, float* Weight1, float* Weight2, float* bias1, float* bias2, size_t input_num, size_t input_dim, size_t hidden_num, size_t class_num, float lr, size_t batch_num);

/**
 * @brief Compute the accuracy rate.
 * 
 * This function calculates the mean accuracy of the predicted results compared to the true labels.
 * 
 * @param result 1D array of predicted class labels, of shape (images_num).
 * @param labels_array 1D array of true class labels, of shape (images_num).
 * @param images_num Number of samples.
 * @param num_classes Number of output classes.
 * @return float Mean accuracy rate.
 */

#endif // NETWORK_HPP
