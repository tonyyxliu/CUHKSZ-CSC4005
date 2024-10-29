#include <iostream>
#include <chrono>
#include "sparse_matrix.hpp"

// compute another column and save it as a dense vector
size_t scatter(const SparseMatrix& A, SparseMatrix& C, int iCol, int beta, int *w, int *x, int mark, size_t nnz){
    for (auto pos = A.start_[iCol]; pos < A.start_[iCol + 1]; ++pos){
        int iRow = A.idx_[pos];
        if (w[iRow] < mark) {
            w[iRow] = mark;
            C.idx_[nnz++] = iRow;
            x[iRow] = beta * A.val_[pos];
        }
        else x[iRow] += beta * A.val_[pos];
    }
    return nnz;
}

SparseMatrix matrix_multiply(const SparseMatrix& A, const SparseMatrix& B) {
    if (A.n_col_ != B.n_row_) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }

    size_t nnz = 0, n_row = A.n_row_, n_col = B.n_col_, nnz_a = A.val_.size(), nnz_b = B.val_.size();
    std::vector<int> w(n_row, 0);
    std::vector<int> x(n_row);

    SparseMatrix C;
    C.n_row_ = n_row;
    C.n_col_ = n_col;
    C.start_.resize(n_col + 1);
    C.ensure_nnz_room(nnz_a + nnz_b, nnz_a + nnz_b);

    for (auto iCol = 0; iCol < n_col; ++iCol){
        C.ensure_nnz_room(nnz + n_row, 2 * C.val_.size() + n_row);
        C.start_[iCol] = nnz;

        // compute the new column and scatter it into a dense vector
        for (auto pos = B.start_[iCol]; pos < B.start_[iCol + 1]; ++pos)
            nnz = scatter(A, C, B.idx_[pos], B.val_[pos], w.data(), x.data(), iCol + 1, nnz);

        // gather dense vector components
        for (auto pos = C.start_[iCol]; pos < nnz; ++pos) {
            C.val_[pos] = x[C.idx_[pos]];
        }
    }

    C.start_[n_col] = nnz;
    C.idx_.resize(nnz);
    C.val_.resize(nnz);
    return C;
}


int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        throw std::invalid_argument(
                "Invalid argument, should be: ./executable "
                "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    const std::string matrix1_path = argv[1];
    const std::string matrix2_path = argv[2];
    const std::string result_path = argv[3];

    SparseMatrix matrix1, matrix2;
    matrix1.loadFromFile(matrix1_path);
    matrix2.loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    SparseMatrix result = matrix_multiply(matrix1, matrix2);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);

    result.save2File(result_path);

    std::cout << "Output file to: " << result_path << std::endl;

    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    return 0;
}
