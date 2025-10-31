//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//

#ifndef CSC4005_PROJECT_3_UTILS_HPP
#define CSC4005_PROJECT_3_UTILS_HPP

#include <fstream>
#include <iostream>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

const int CSC4005_SEED = 4005;  // Global seed to maintain consistent RNG


std::vector<int> loadVectorFromFile(const std::string& filename);

void saveVectorToFile(const std::vector<int> vec, const std::string& filename);

std::vector<int> createUniformVec(int size, int seed = CSC4005_SEED);

void checkSortResult(std::vector<int>& vec1, std::vector<int>& vec2);

void checkSearchResult(const std::vector<int>& vec, 
                       const std::vector<int>& search_targets,
                       const std::vector<int>& results);

std::vector<int> createCuts(int start, int end, int tasks_num);

void print_vec(std::vector<int> &vec, int low, int high);

std::vector<int> prefixSum(std::vector<int> &data);

#endif  // CSC4005_PROJECT_3_UTILS_HPP
