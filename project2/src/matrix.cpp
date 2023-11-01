//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Simple Matrix Implementation
//

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <random>
#include <memory.h>
#include "matrix.hpp"

Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
    // Allocate memory for the matrix
    data = new int*[rows];
    for (size_t i = 0; i < rows; ++i) {
        // +8 for SIMD convenience
        data[i] = new int[cols + 8];
        memset(data[i], 0, cols * sizeof(int));
    }
}

Matrix::~Matrix() {
    // Destructor to free memory
    if (data != nullptr) {
        for (size_t i = 0; i < rows; ++i) {
            delete[] data[i];
        }
        delete[] data;
    }
}

int* Matrix::operator[](size_t rowIndex) {
    if (rowIndex < rows) {
        return data[rowIndex];
    } else {
        throw std::out_of_range("Row index out of range");
    }
}

const int* Matrix::operator[](size_t rowIndex) const {
    if (rowIndex < rows) {
        return data[rowIndex];
    } else {
        throw std::out_of_range("Row index out of range");
    }
}

void Matrix::display() const {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << data[i][j] << ' ';
        }
        std::cout << std::endl;
    }
}

size_t Matrix::getRows() const { return rows; }

size_t Matrix::getCols() const { return cols; }

Matrix Matrix::loadFromFile(const std::string& filename) {
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    size_t rows, cols;
    inputFile >> rows >> cols;

    Matrix loadedMatrix(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (!(inputFile >> loadedMatrix[i][j])) {
                throw std::runtime_error("Error reading data from file: " +
                                         filename);
            }
        }
    }

    inputFile.close();

    return loadedMatrix;
}

void Matrix::saveToFile(const std::string& filename) const {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " +
                                 filename);
    }

    outputFile << rows << ' ' << cols << std::endl;

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            outputFile << data[i][j] << ' ';
        }
        outputFile << std::endl;
    }

    outputFile.close();
}

Matrix::Matrix(Matrix&& other) noexcept {
    data = other.data;
    rows = other.rows;
    cols = other.cols;
    other.data = nullptr;
    other.rows = 0;
    other.cols = 0;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    // prevent self-assignment
    if (this == &other) {
        return *this;
    }
    // Free the memory of the current object
    if (data != nullptr) {
        for (size_t i = 0; i < rows; ++i) {
            delete[] data[i];
        }
        delete[] data;
    }
    // Move the data from the other object
    data = other.data;
    rows = other.rows;
    cols = other.cols;
    // Reset the other object
    other.data = nullptr;
    other.rows = 0;
    other.cols = 0;
    return *this;
}
