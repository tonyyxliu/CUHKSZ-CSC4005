#include "simple_ml_openacc.hpp"

int main()
{
    DataSet *train_data = parse_mnist("./dataset/training/train-images.idx3-ubyte",
                                      "./dataset/training/train-labels.idx1-ubyte");
    DataSet *test_data = parse_mnist("./dataset/testing/t10k-images.idx3-ubyte",
                                     "./dataset/testing/t10k-labels.idx1-ubyte");

    std::cout << "Training softmax regression (GPU)" << std::endl;
    train_softmax_openacc(train_data, test_data, 10, 10, 0.2);

    delete train_data;
    delete test_data;

    return 0;
}
