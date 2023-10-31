//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//

#ifndef CSC4005_PROJECT_3_UTILS_HPP
#define CSC4005_PROJECT_3_UTILS_HPP

#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

std::vector<int> loadVectorFromFile(const std::string& filename);

void saveVectorToFile(const std::vector<int> vec, const std::string& filename);

std::vector<int> createRandomVec(int size, int seed);

void checkSortResult(std::vector<int>& vec1, std::vector<int>& vec2);

std::vector<int> createCuts(int start, int end, int tasks_num);

void print_vec(std::vector<int> &vec, int low, int high);

#endif