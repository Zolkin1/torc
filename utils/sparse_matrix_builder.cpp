//
// Created by zolkin on 1/3/24.
//

#include "sparse_matrix_builder.h"

namespace torc::utils {
    using triplet_t = Eigen::Triplet<double>;
    SparseMatrixBuilder::SparseMatrixBuilder() = default;

    void SparseMatrixBuilder::Reserve(const int num_nz) {
        triplet_.erase(triplet_.begin(), triplet_.end());
        triplet_.reserve(num_nz);
    }

    void SparseMatrixBuilder::SetDiagonalMatrix(const double val, const int row_start, const int col_start, const int num_diag) {
        for (int i = 0; i < num_diag; i++) {
            triplet_.emplace_back( row_start + i, col_start + i,val);
        }
    }

    void SparseMatrixBuilder::SetFromMatrix(const matrix_t& M, const int row_start, const int col_start) {
        for (int i = 0; i < M.rows(); i++) {
            for (int j = 0; j < M.cols(); j++) {
                if (M(i,j) != 0) {  // TODO: Consider making it less than an epsilon
                    triplet_.emplace_back(row_start + i ,col_start + j, M(i,j));
                }
            }
        }
    }

    void SparseMatrixBuilder::SetVectorDiagonally(const vector_t& vec, const int row_start, const int col_start) {
        for (int i = 0; i < vec.size(); i++) {
            if (vec(i) != 0) {
                triplet_.emplace_back(row_start + i, col_start + i, vec(i));
            }
        }
    }

    const std::vector<triplet_t>& SparseMatrixBuilder::GetTriplet() const {
        return triplet_;
    }

    void SparseMatrixBuilder::SetSparseMatrix(sp_matrix_t &sp) const {
        sp.setFromTriplets(triplet_.cbegin(), triplet_.cend());
    }
}
