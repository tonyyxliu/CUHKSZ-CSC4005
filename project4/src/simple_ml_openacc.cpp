#include "simple_ml_openacc.hpp"

void matrix_dot_openacc(const float *A, const float *B,
                        float *C, size_t m, size_t n, size_t k)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

void matrix_dot_trans_openacc(const float *A, const float *B, float *C, size_t n, size_t m, size_t k)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

void matrix_trans_dot_openacc(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

void matrix_minus_openacc(float *A, const float *B, size_t m, size_t n)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

void matrix_mul_scalar_openacc(float *C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

void matrix_div_scalar_openacc(float *C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

void matrix_softmax_normalize_openacc(float *C, size_t m, size_t n)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

void vector_to_one_hot_matrix_openacc(const unsigned char *y, float *Y, size_t m, size_t k)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

void softmax_regression_epoch_openacc(const float *X, const unsigned char *y,
                                      float *theta, size_t m, size_t n, size_t k,
                                      float lr, size_t batch)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

void train_softmax_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t epochs, float lr, size_t batch)
{
    /*
    Example function to fully train a softmax regression classifier
    */
    size_t size = train_data->input_dim * num_classes;
    float *theta = new float[size];
    memset(theta, 0, size * sizeof(float));
    size_t size_tr = train_data->images_num * num_classes;
    size_t size_te = test_data->images_num * num_classes;
    float *train_result = new float[size_tr];
    float *test_result = new float[size_te];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    std::chrono::milliseconds elapsed_time;
    // BEGIN YOUR CODE
  
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        train_loss = mean_softmax_loss_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
  
   // END YOUR CODE
    delete[] theta;
    delete[] train_result;
    delete[] test_result;
}

float mean_softmax_loss_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

float mean_err_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

void matrix_mul_openacc(float *A, const float *B, size_t size)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

void nn_epoch_openacc(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float lr, size_t batch)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

void train_nn_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t hidden_dim, size_t epochs, float lr, size_t batch)
{
    size_t size_w1 = train_data->input_dim * hidden_dim;
    size_t size_w2 = hidden_dim * num_classes;
    float *W1 = new float[size_w1];
    float *W2 = new float[size_w2];
    std::mt19937 rng;
    rng.seed(0);
    std::normal_distribution<float> dist(0.0, 1.0);
    for (size_t i = 0; i < size_w1; i++)
    {
        W1[i] = dist(rng);
    }
    for (size_t i = 0; i < size_w2; i++)
    {
        W2[i] = dist(rng);
    }
    matrix_div_scalar(W1, sqrtf(hidden_dim), train_data->input_dim, hidden_dim);
    matrix_div_scalar(W2, sqrtf(num_classes), hidden_dim, num_classes);
    size_t size_tr = train_data->images_num * num_classes;
    size_t size_te = test_data->images_num * num_classes;
    float *train_result = new float[size_tr];
    float *test_result = new float[size_te];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    std::chrono::milliseconds elapsed_time;
    // BEGIN YOUR CODE
  
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        train_loss = mean_softmax_loss_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
  
    // END YOUR CODE
    delete[] W1;
    delete[] W2;
    delete[] train_result;
    delete[] test_result;
}
