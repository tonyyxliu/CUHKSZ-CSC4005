#include "utils.hpp"
#include "mlp_network.hpp"

int main(int argc, char* argv[])
{
    std::string path_train_data_x(argv[1]);
    std::string path_train_data_y(argv[2]);
    std::string path_test_data_x(argv[3]);
    std::string path_test_data_y(argv[4]);

    int num_classes = 10;
    int hidden_dim = std::stoi(argv[5]);
    int epochs = std::stoi(argv[6]);
    float learning_rate = std::stof(argv[7]);
    int batch = std::stoi(argv[8]);

    DataSet* train_data = parse_mnist(path_train_data_x,
                                      path_train_data_y);
    DataSet* test_data = parse_mnist(path_test_data_x,
                                     path_test_data_y);
    std::cout << "Training two layer neural network "<< hidden_dim <<" hidden units" << std::endl;
    train_nn(train_data, test_data, num_classes, hidden_dim, epochs, learning_rate, batch);

    delete train_data;
    delete test_data;

    return 0;
}
