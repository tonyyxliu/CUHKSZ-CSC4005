#include "simple_ml_openacc.hpp"

int main(int argc, char *argv[])
{
    std::string path_train_data_x(argv[1]);
    std::string path_train_data_y(argv[2]);
    std::string path_test_data_x(argv[3]);
    std::string path_test_data_y(argv[4]);

    DataSet *train_data = parse_mnist(path_train_data_x,
                                      path_train_data_y);
    DataSet *test_data = parse_mnist(path_test_data_x,
                                     path_test_data_y);

    std::cout << "Training two layer neural network w/ 400 hidden units (GPU)" << std::endl;
    train_nn_openacc(train_data, test_data, 10, 400, 20, 0.2);

    delete train_data;
    delete test_data;

    return 0;
}
