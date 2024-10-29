#include <fstream>
#include "sparse_matrix.hpp"

SparseMatrix::SparseMatrix(): start_(1, 0), idx_(0), val_(0), n_col_(0), n_row_(0) {}

bool SparseMatrix::operator==(SparseMatrix &&rhs) const{
    return ((n_col_ == rhs.n_col_) &&
            (n_row_ == rhs.n_row_) &&
            (start_ == rhs.start_) &&
            (idx_ == rhs.idx_) &&
            (val_ == rhs.val_));
}

bool SparseMatrix::operator==(const SparseMatrix &rhs) const {
    return ((n_col_ == rhs.n_col_) &&
            (n_row_ == rhs.n_row_) &&
            (start_ == rhs.start_) &&
            (idx_ == rhs.idx_) &&
            (val_ == rhs.val_));
}

void SparseMatrix::ensure_nnz_room(size_t min_nnz, size_t rec_nnz) {
    if (idx_.size() < min_nnz){
        if (rec_nnz < min_nnz)
            throw std::logic_error("recommended room " + std::to_string(rec_nnz) +
                                   " is less than minimum required " + std::to_string(min_nnz));
        try{
            idx_.resize(rec_nnz);
            val_.resize(rec_nnz);
        }
        catch (std::bad_alloc& ){
            throw std::length_error("can't reallocate matrix with " + std::to_string(rec_nnz) + " elements");
        }
    }
}


void SparseMatrix::loadFromFile(const std::string &filename) {
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    size_t nnz;
    inputFile >> n_col_ >> n_row_ >> nnz;
    idx_.resize(nnz);
    val_.resize(nnz);
    start_.resize(n_col_ + 1);

    for (auto iCol = 0; iCol <= n_col_; ++iCol) inputFile >> start_[iCol];
    for (auto it = 0; it < nnz; ++it) inputFile >> idx_[it];
    for (auto it = 0; it < nnz; ++it) inputFile >> val_[it];

    inputFile.close();
}

void SparseMatrix::save2File(const std::string &filename) const {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    outputFile << n_col_ << "\t" << n_row_ << "\t" << idx_.size() << std::endl;

    for (auto& it: start_) outputFile << it << "\t";
    outputFile << std::endl;
    for (auto& it: idx_) outputFile << it << "\t";
    outputFile << std::endl;
    for (auto& it: val_) outputFile << it << "\t";
    outputFile << std::endl;


    outputFile.close();
}
