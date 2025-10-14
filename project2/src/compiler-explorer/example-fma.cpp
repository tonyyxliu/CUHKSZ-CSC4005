/**
 * Copy and Paste the content to https://godbolt.org/
 */

#include <stdexcept>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <random>
#include <regex>
#include <memory.h>

#define MAT_DATATYPE double

class Matrix
{
private:
    MAT_DATATYPE* data;
    size_t rows;
    size_t cols;

public:
    // Constructor
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols)
    {
        // Allocate memory for the matrix
        data = new MAT_DATATYPE[rows * cols]();
    }

    // Destructor
    ~Matrix()
    {
        if (data != nullptr)
        {
            delete[] data;
        }
    }

    MAT_DATATYPE* getData() { return data; }

    const MAT_DATATYPE* getDataConst() const { return data; }

    // Overload the [] operator for convenient element access
    MAT_DATATYPE& operator[](size_t idx);

    // Read only element access
    const MAT_DATATYPE& operator[](size_t idx) const;

    MAT_DATATYPE& operator()(size_t rowIdx, size_t colIdx)
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

    const MAT_DATATYPE& operator()(size_t rowIdx, size_t colIdx) const
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

    // Get the row numbers of a matrix
    size_t getRows() const { return rows; }

    // Get the column numbers of a matrix
    size_t getCols() const { return cols; }

    /**
     * Check whether two input matrices are completely identical
     *
     * Use 1e-8 as episilon for double-precision equality checking
     *
     * @return true if identical, false otherwise
     */
    static bool isIdentical(const Matrix& mat1, const Matrix& mat2,
                            double epsilon = 1e-12)
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

    /**
     * Load a matrix from a file
     * @param filename: path to the matrix .txt
     */
    static Matrix loadFromFile(const std::string& filename)
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

    /**
     * Find the matrix No. through Regexp searching
     * @param filename: path of the matrix txt file
     * @return mat No.
     */
    static int getMatNumber(const std::string& mat_path)
    {
        std::regex pattern(R"(matrix(\d)\.txt)");
        std::smatch matches;
        if (std::regex_search(mat_path, matches, pattern) && matches.size() > 1)
        {
            return std::stoi(matches[1].str());
        }
        return -1; // invalid file path, unable to find matrix number
    }

    /**
     * Get the result matrix for veriication
     */
    static Matrix getResultMatrix(const std::string& mat1_path,
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
            std::string replacement =
                "matrix" + std::to_string(min_mat_number) + "x" +
                std::to_string(max_mat_number) + "_result.txt";
            const std::string result_path =
                std::regex_replace(mat1_path, pattern, replacement);
            return Matrix::loadFromFile(result_path);
        }
        std::cerr << "Invalid matrix path: " << mat1_path << "\n";
        return Matrix(0, 0); // return size = 0 matrix if invalid
    }
};

/**
 * Naive Matmul with FMA enabled by declaring a __restrict__ ptr
 */
Matrix matrix_multiply_fma(const Matrix& matrix1, const Matrix& matrix2)
{
    if (matrix1.getCols() != matrix2.getRows())
    {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
    std::cout << "M = " << M << ", N = " << N << ", K = " << K << std::endl;

    Matrix result(M, N);

    // Mark that result_data ptr cannot be modified elsewhere
    MAT_DATATYPE* __restrict__ result_data = result.getData();

    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            const size_t result_idx = i * N + j;
            for (size_t k = 0; k < K; ++k)
            {
                result_data[result_idx] += matrix1(i, k) * matrix2(k, j);
            }
        }
    }

    return result;
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        throw std::invalid_argument("Invalid argument, should be: ./executable "
                                    "/path/to/matrix1 /path/to/matrix2\n");
    }

    const std::string matrix1_path = argv[1];
    const std::string matrix2_path = argv[2];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);
    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_fma(matrix1, matrix2);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    Matrix ground_truth = Matrix::getResultMatrix(matrix1_path, matrix2_path);
    std::cout << "Verification: "
              << ((Matrix::isIdentical(result, ground_truth)) ? "Passed"
                                                              : "Failed")
              << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;
    return 0;
}
