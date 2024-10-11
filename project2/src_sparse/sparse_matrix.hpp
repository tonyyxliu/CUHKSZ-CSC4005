#ifndef SPARSE_SPARSE_MATRIX_H
#define SPARSE_SPARSE_MATRIX_H

#include <vector>

class SparseMatrix{
public:

    SparseMatrix();

    SparseMatrix& operator=(const SparseMatrix& rhs) = default;
    SparseMatrix& operator=(SparseMatrix&& rhs) = default;
    SparseMatrix(const SparseMatrix& rhs) = default;
    SparseMatrix(SparseMatrix&& rhs) =default;
    ~SparseMatrix() = default;
    bool operator==(const SparseMatrix& rhs) const;
    bool operator==(SparseMatrix&& rhs) const;

    void ensure_nnz_room(size_t min_nnz, size_t rec_nnz);

    void loadFromFile(const std::string& filename);
    void save2File(const std::string& filename) const;

    std::vector<size_t> start_;
    std::vector<int> idx_;
    std::vector<int> val_;
    size_t n_col_;
    size_t n_row_;
};

#endif //SPARSE_SPARSE_MATRIX_H
