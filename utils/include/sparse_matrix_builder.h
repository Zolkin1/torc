//
// Created by zolkin on 1/3/24.
//

#ifndef SPARSE_MATRIX_BUILDER_H
#define SPARSE_MATRIX_BUILDER_H

#include <eigen3/Eigen/SparseCore>

namespace torc::utils {
    class SparseMatrixBuilder {
        using triplet_t = Eigen::Triplet<double>;
        using vector_t = Eigen::VectorXd;
        using matrix_t = Eigen::MatrixXd;
        using sp_matrix_t = Eigen::SparseMatrix<double>;

    public:
        SparseMatrixBuilder();

        void Reserve(int num_nz);

        void SetDiagonalMatrix(double val, int row_start, int col_start, int num_diag);

        void SetFromMatrix(const matrix_t& M, int row_start, int col_start);

        void SetVectorDiagonally(const vector_t& vec, int row_start, int col_start);

        [[nodiscard]] const std::vector<triplet_t>& GetTriplet() const;

        void SetSparseMatrix(sp_matrix_t& sp) const;

    private:
        std::vector<triplet_t> triplet_;
    };
}


#endif // SPARSE_MATRIX_BUILDER_H