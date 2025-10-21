//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Modified by Liu Yuxuan on 2025/10/12
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Simple Matrix Declaration
//

#ifndef CSC4005_PROJECT_2_MATRIX_HPP
#define CSC4005_PROJECT_2_MATRIX_HPP

#include <iostream>
#include <vector>
#include <regex>

#define MAT_DATATYPE double

class Matrix
{
private:
    MAT_DATATYPE* data;
    size_t rows;
    size_t cols;

public:
    // Constructor
    Matrix(size_t rows, size_t cols);

    // Destructor
    ~Matrix();

    void release();

    MAT_DATATYPE* getData();

    const MAT_DATATYPE* getDataConst() const;

    // Overload the [] operator for convenient element access
    MAT_DATATYPE& operator[](size_t idx);

    // Read only element access
    const MAT_DATATYPE& operator[](size_t idx) const;

    MAT_DATATYPE& operator()(size_t rowIdx, size_t colIdx);

    const MAT_DATATYPE& operator()(size_t rowIdx, size_t colIdx) const;

    // Function to display the matrix
    void display() const;

    // Get the row numbers of a matrix
    size_t getRows() const;

    // Get the column numbers of a matrix
    size_t getCols() const;

    /**
     * Get blocked data
     */
    void getBlock(MAT_DATATYPE* __restrict__ block_data, size_t row_start,
                  size_t col_start, size_t block_size) const;

    /**
     * Set blocked data
     * [Note]: You may add another 'incrementBlock' method if your result block
     * is set multiple times
     */
    void setBlock(const MAT_DATATYPE* const block_data, size_t row_start,
                  size_t col_start, size_t block_size);

    // disable copy
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;

    // enable move
    Matrix(Matrix&&) noexcept;
    Matrix& operator=(Matrix&&) noexcept;

    // Save a matrix to a file
    void saveToFile(const std::string& filename) const;

    /**
     * Check whether two input matrices are completely identical
     *
     * Use 1e-12 as episilon for double-precision equality checking
     *
     * @return true if identical, false otherwise
     */
    static bool isIdentical(const Matrix& mat1, const Matrix& mat2,
                            double epsilon = 0.5);

    /**
     * Load a matrix from a file
     * @param filename: path to the matrix .txt
     */
    static Matrix loadFromFile(const std::string& filename);

    /**
     * Find the matrix No. through Regexp searching
     * @param filename: path of the matrix txt file
     * @return mat No.
     */
    static int getMatNumber(const std::string& mat_path);

    /**
     * Get the result matrix for veriication
     */
    static Matrix getResultMatrix(const std::string& mat1_path,
                                  const std::string& mat2_path);
};

#endif
