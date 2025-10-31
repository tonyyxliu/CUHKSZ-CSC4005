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

void checkSearchResult(const std::vector<int>& vec, 
                       const std::vector<int>& search_targets,
                       const std::vector<int>& results) {
    // Check if results vector size matches search_targets size
    if (results.size() != search_targets.size()) {
        std::cout << "Fail to pass the searching result check!" << std::endl;
        std::cout << "The size of the results vector is expected to be " << search_targets.size() << std::endl;
        std::cout << "But your results vector's size is " << results.size() << std::endl; 
        return;
    }
    
    // Use std::lower_bound as the standard for verification (finds FIRST occurrence)
    auto standardBinarySearch = [](const std::vector<int>& arr, int target) -> int {
        auto it = std::lower_bound(arr.begin(), arr.end(), target);
        if (it != arr.end() && *it == target) {
            return std::distance(arr.begin(), it);
        }
        return -1;
    };
    
    // Check each search result
    int error_count = 0;
    const int MAX_ERRORS_TO_SHOW = 5;
    
    for (int i = 0; i < search_targets.size(); ++i) {
        int target = search_targets[i];
        int student_result = results[i];
        int expected_result = standardBinarySearch(vec, target);
        
        // Compare with std::lower_bound result
        if (student_result != expected_result) {
            if (error_count < MAX_ERRORS_TO_SHOW) {
                std::cout << "Fail to pass the searching result check!" << std::endl;
                std::cout << "For search target " << target << " (index " << i << "):" << std::endl;
                std::cout << "  Your result: " << student_result;
                if (student_result == -1) {
                    std::cout << " (not found)";
                } else if (student_result >= 0 && student_result < vec.size()) {
                    std::cout << " (vec[" << student_result << "] = " << vec[student_result] << ")";
                } else {
                    std::cout << " (out of bounds)";
                }
                std::cout << std::endl;
                std::cout << "  Expected (std::lower_bound): " << expected_result;
                if (expected_result == -1) {
                    std::cout << " (not found)";
                } else {
                    std::cout << " (vec[" << expected_result << "] = " << vec[expected_result] << ")";
                }
                std::cout << std::endl;
            }
            error_count++;
        }
        
        if (error_count > 0 && error_count == MAX_ERRORS_TO_SHOW) {
            std::cout << "... (showing first " << MAX_ERRORS_TO_SHOW << " errors)" << std::endl;
        }
    }
    
    if (error_count > 0) {
        std::cout << "Total errors found: " << error_count << " out of " << search_targets.size() << " searches" << std::endl;
        std::cout << "Fail to pass the searching result check!" << std::endl;
    } else {
        std::cout << "Pass the searching result check!" << std::endl;
    }
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
