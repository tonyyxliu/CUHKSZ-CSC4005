//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
// 
// Modified by Liu Yuxuan on 2024/10/27
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Utility functions for Project 3: Parallel Sorting
//

#include "utils.hpp"

DistType str_2_dist_type(const std::string& str) {
    if (!strcasecmp(str.c_str(), "uniform"))
        return Uniform;
    else if (!strcasecmp(str.c_str(), "normal"))
        return Normal;
    else
        throw std::invalid_argument("dist_type option can only be 'uniform' or 'normal'\n");
}

std::vector<int> genRandomVec(int size, DistType dist, int seed) {
    switch (dist) {
        case Uniform:
            return createUniformVec(size, seed);
        case Normal:
            return createNormalVec(size, seed);
    }
}

std::vector<int> createUniformVec(int size, int seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> distribution(0, size);
    std::vector<int> randomVector;
    randomVector.reserve(size);
    for (int i = 0; i < size; i++) {
        int randomValue = distribution(gen);
        randomVector.push_back(randomValue);
    }
    return randomVector;
}

std::vector<int> createNormalVec(int size, int seed) {
    std::mt19937 gen(seed);
    // According to the empirical law of normal distribution
    // Around 68% of data appear within [mean - stddev, mean + stddev]
    // Around 95% of data appear within [mean - 2 * stddev, mean + 2 * stddev]
    // Around 99.7% of data appear within [mean - 3 * stddev, mean + 3 * stddev]
    double mean = size / 2.0;
    double stddev = size / 10.0;
    std::normal_distribution<> distribution(mean, stddev);
    std::vector<int> randomVector;
    randomVector.reserve(size);
    for (int i = 0; i < size; i++) {
        int num;
        do {
            num = static_cast<int>(distribution(gen));
        } while (num < 0 || num > size); // Ensure within the range
        randomVector.push_back(num);
    }
    return randomVector;
}

void checkSortResult(std::vector<int>& vec1, std::vector<int>& vec2) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::sort(vec1.begin(), vec1.end());
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cout << "std::sort Time: " << elapsed_time.count() << " milliseconds"
            << std::endl;

    if (vec1.size() != vec2.size()) {
        std::cout << "Fail to pass the sorting result check!" << std::endl;
        std::cout << "The size of the sorted vector is expected to be " << vec1.size() << std::endl;
        std::cout << "But your sorted vector's size is " << vec2.size() << std::endl; 
        return;
    }
    for (int i = 0; i < vec1.size(); ++i) {
        if (vec1[i] != vec2[i]) {
            std::cout << "Fail to pass the sorting result check!" << std::endl;
            std::cout << i << "th element of the sorted vector is expected to be " << vec1[i] << std::endl;
            std::cout << "But your " << i << "th element is " << vec2[i] << std::endl; 
            return;
        }
    }
    std::cout << "Pass the sorting result check!" << std::endl;
}

std::vector<int> createCuts(int start, int end, int tasks_num) {
    std::vector<int> cuts(tasks_num + 1, start);
    int data_per_task = (end + 1 - start) / tasks_num;
    int left_data_num = (end + 1 - start) % tasks_num;
    int divided_left_data_num = 0;

    for (int i = 0; i < tasks_num; i++) {
        if (divided_left_data_num < left_data_num) {
            cuts[i + 1] = cuts[i] + data_per_task + 1;
            divided_left_data_num++;
        } else
            cuts[i + 1] = cuts[i] + data_per_task;
    }
    return cuts;
}

std::vector<int> loadVectorFromFile(const std::string& filename) {
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    std::vector<int> vec;
    int data;
    while (inputFile >> data) {
        vec.push_back(data);
    }
    return vec;
}

void saveVectorToFile(const std::vector<int> vec, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " +
                                 filename);
    }
    for (int data: vec) {
        outputFile << data << ' ';
    }
    outputFile << std::endl;
    outputFile.close();
}

void print_vec(std::vector<int> &vec, int low, int high) {
    for (int i = low; i < high; ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

// Compute the displacement vector for indexing
std::vector<int> prefixSum(std::vector<int> &data)
{
	std::vector<int> output(data.size(), 0);
	int sum = 0;
	for (int i = 0; i < data.size(); i++)
	{
		output[i] = sum;
		sum += data[i];
	}
	return output;
}
