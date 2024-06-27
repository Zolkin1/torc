//
// Created by gavin on 6/27/24.
//

#ifndef CONSTRAINT_DATA_H
#define CONSTRAINT_DATA_H

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

namespace torc::constraint {
    enum CONSTRAINT_T {
        Equals,         // equality constraint
        LessThan,       // less than or equal to constraint
        GreaterThan     // greater than or equal to constraint
    };

    /**
     * Holds the matrices that the Constraint class will use to return values.
     * @tparam scalar_t type of scalar to use in the constraint
     */
    template <class scalar_t>
    struct ConstraintData {
        Eigen::MatrixX<scalar_t> ineq_grad;
        Eigen::SparseMatrix<scalar_t> ineq_grad_sparse;
        Eigen::MatrixX<scalar_t> eq_grad;
        Eigen::SparseMatrix<scalar_t> eq_grad_sparse;
        Eigen::VectorX<scalar_t> bound_low;
        Eigen::VectorX<scalar_t> bound_high;
        Eigen::VectorX<scalar_t> bound_all;
        Eigen::VectorX<scalar_t> bound_eq;
        std::vector<CONSTRAINT_T> types;
        std::vector<size_t> reps;
    };
}

#endif //CONSTRAINT_DATA_H
