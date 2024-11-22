#include "ops.hpp"
const float epsilon = 1e-20;

void gemm(const float* A, const float* B, float* Out, size_t batch, size_t mn, size_t k)
{
    // BEGIN YOUR CODE HERE ->

    // END YOUR CODE HERE <-
}

void add_bias(float* A, float* B, const float* bias, size_t batch, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    
    // END YOUR CODE HERE <-
}

void Relu(float* A, float* B, size_t size)
{
    // BEGIN YOUR CODE HERE ->
    
    // END YOUR CODE HERE <-
}

void Softmax(float* A, float* B, size_t batch, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    
    // END YOUR CODE HERE <-
}

void vector_to_one_hot_matrix(const unsigned char* A, float* B, size_t batch, size_t out_dim)
{
    memset(B, 0, batch * out_dim * sizeof(float)); // initialize to 0
    for (size_t i = 0; i < batch; i++)
    {
        B[i * out_dim + A[i]] = 1;
    }
}

void cross_entropy_loss(const float* A, const float* B, float* Loss, size_t batch, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
        // Optional for your debug    
    // END YOUR CODE HERE <-
}

void cross_entropy_loss_grad(const float* A, const float* B, float* Grad, size_t batch, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    
    // END YOUR CODE HERE <-
}

void update_bias(float* Bias, const float* Output_Loss, size_t batch, float lr, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    
    // END YOUR CODE HERE <-
}

void input_grad(const float* Weight, const float* Output_Loss, float* Input, float* Grad, size_t batch, size_t in_dim, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    
    // END YOUR CODE HERE <-
}

void update_weight(float* Weight, const float* Output_Loss, const float* Input, size_t batch, float lr, size_t in_dim, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    
    // END YOUR CODE HERE <-
}

void relu_grad(const float* A, float* Grad, size_t batch, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    
    // END YOUR CODE HERE <-
}