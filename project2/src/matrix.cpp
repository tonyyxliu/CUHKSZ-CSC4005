//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Modified by Liu Yuxuan on 2025/10/12
// Email: yuxuanliu1@link.cuhk.edu.cn
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

Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols)
{
    // Allocate memory for the matrix
    data = new MAT_DATATYPE[rows * cols]();
}

Matrix::~Matrix()
{
    // Destructor to free memory
    if (data != nullptr)
    {
        delete[] data;
    }
}

void Matrix::release()
{
    if (data != nullptr)
    {
        delete[] data;
        data = nullptr;
        rows = 0;
        cols = 0;
    }
}

MAT_DATATYPE* Matrix::getData() { return data; }

const MAT_DATATYPE* Matrix::getDataConst() const { return data; }

size_t Matrix::getRows() const { return rows; }

size_t Matrix::getCols() const { return cols; }

MAT_DATATYPE& Matrix::operator[](size_t idx)
{
    if (idx < rows * cols)
        return data[idx];
    else
        throw std::out_of_range("Row index out of range");
}

const MAT_DATATYPE& Matrix::operator[](size_t idx) const
{
    if (idx < rows * cols)
        return data[idx];
    else
        throw std::out_of_range("Row index out of range");
}

MAT_DATATYPE& Matrix::operator()(size_t rowIdx, size_t colIdx)
{
    if (rowIdx < rows && colIdx < cols)
    {
        return data[rowIdx * cols + colIdx];
    }
    else
    {
        throw std::out_of_range("Row index out of range");
    }
}

const MAT_DATATYPE& Matrix::operator()(size_t rowIdx, size_t colIdx) const
{
    if (rowIdx < rows && colIdx < cols)
    {
        return data[rowIdx * cols + colIdx];
    }
    else
    {
        throw std::out_of_range("Row index out of range");
    }
}

void Matrix::getBlock(MAT_DATATYPE* __restrict__ block_data, size_t row_start,
                      size_t col_start, size_t block_size) const
{
    const MAT_DATATYPE* const data = getDataConst();
    const size_t num_cols = getCols();
    for (size_t i = 0; i < block_size; ++i)
    {
        const size_t base_idx = (row_start + i) * num_cols;
        const size_t block_base_idx = i * block_size;
#pragma GCC ivdep
#pragma GCC vector always
#pragma GCC unroll 8
        for (size_t j = 0; j < block_size; ++j)
        {
            block_data[block_base_idx + j] = data[base_idx + j];
        }
    }
}

void Matrix::setBlock(const MAT_DATATYPE* const block_data, size_t row_start,
                      size_t col_start, size_t block_size)
{
    const size_t num_cols = getCols();
    for (size_t i = 0; i < block_size; ++i)
    {
        const size_t base_idx = (row_start + i) * num_cols;
        const size_t block_base_idx = i * block_size;
#pragma GCC ivdep
#pragma GCC vector always
#pragma GCC unroll 8
        for (size_t j = 0; j < block_size; ++j)
        {
            data[base_idx + j] += block_data[block_base_idx + j];
        }
    }
}

void Matrix::display() const
{
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            std::cout << data[i * cols + j] << ' ';
        }
        std::cout << std::endl;
    }
}

void Matrix::saveToFile(const std::string& filename) const
{
    std::ofstream outputFile(filename);
    if (!outputFile.is_open())
    {
        throw std::runtime_error("Failed to open file for writing: " +
                                 filename);
    }

    outputFile << rows << ' ' << cols << std::endl;

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            outputFile << data[i * cols + j] << ' ';
        }
        outputFile << std::endl;
    }

    outputFile.close();
}

Matrix::Matrix(Matrix&& other) noexcept
{
    data = other.data;
    rows = other.rows;
    cols = other.cols;
    other.data = nullptr;
    other.rows = 0;
    other.cols = 0;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept
{
    // prevent self-assignment
    if (this == &other)
    {
        return *this;
    }
    // Free the memory of the current object
    if (data != nullptr) delete[] data;
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

bool Matrix::isIdentical(const Matrix& mat1, const Matrix& mat2, double epsilon)
{
    if (mat1.getRows() != mat2.getRows()) return false;
    if (mat2.getCols() != mat2.getCols()) return false;
    for (size_t i = 0; i < mat1.getRows(); ++i)
    {
        for (size_t j = 0; j < mat1.getCols(); ++j)
        {
            if (std::fabs(mat1(i, j) - mat2(i, j)) < epsilon)
            {
                std::cout << "Mat1 value: " << mat1(i, j) << "\n";
                std::cout << "Mat2 value: " << mat2(i, j) << "\n";
                return false;
            }
        }
    }
    return true;
}

Matrix Matrix::loadFromFile(const std::string& filename)
{
    std::ifstream inputFile(filename);
    if (!inputFile.is_open())
        throw std::runtime_error("Failed to open file: " + filename);

    size_t rows, cols;
    inputFile >> rows >> cols;

    Matrix loadedMatrix(rows, cols);

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            if (!(inputFile >> loadedMatrix(i, j)))
            {
                throw std::runtime_error("Error reading data from file: " +
                                         filename);
            }
        }
    }
    inputFile.close();
    return loadedMatrix;
}

int Matrix::getMatNumber(const std::string& mat_path)
{
    std::regex pattern(R"(matrix(\d)\.txt)");
    std::smatch matches;
    if (std::regex_search(mat_path, matches, pattern) && matches.size() > 1)
    {
        return std::stoi(matches[1].str());
    }
    return -1; // invalid file path, unable to find matrix number
}

Matrix Matrix::getResultMatrix(const std::string& mat1_path,
                               const std::string& mat2_path)
{
    // Get matrix numbers
    int mat1_number = Matrix::getMatNumber(mat1_path);
    int mat2_number = Matrix::getMatNumber(mat2_path);
    int min_mat_number = std::min(mat1_number, mat2_number);
    int max_mat_number = std::max(mat1_number, mat2_number);

    // Replace "matrixA.txt" to "matrixAxB_result.txt"
    std::regex pattern(R"(matrix(\d)\.txt)");
    std::smatch matches;
    if (std::regex_search(mat1_path, matches, pattern))
    {
        std::string replacement = "matrix" + std::to_string(min_mat_number) +
                                  "x" + std::to_string(max_mat_number) +
                                  "_result.txt";
        const std::string result_path =
            std::regex_replace(mat1_path, pattern, replacement);
        return Matrix::loadFromFile(result_path);
    }
    std::cerr << "Invalid matrix path: " << mat1_path << "\n";
    return Matrix(0, 0); // return size = 0 matrix if invalid
}
