//
// Created by gavin on 6/27/24.
//

#ifndef CONSTRAINT_DATA_H
#define CONSTRAINT_DATA_H

#include <eigen3/Eigen/Dense>

namespace torc::constraint {
    enum CONSTRAINT_T {
        Equals,         // equality constraint
        LessThan,       // less than or equal to constraint
        GreaterThan     // greater than or equal to constraint
    };

    template <class scalar_t>
    struct ConstraintData {
        Eigen::MatrixX<scalar_t> inequality_gradients;
        Eigen::MatrixX<scalar_t> equality_gradients;
        Eigen::VectorX<scalar_t> inequality_bound_low;
        Eigen::VectorX<scalar_t> inequality_bound_high;
        Eigen::VectorX<scalar_t> equality_bound;
        std::vector<CONSTRAINT_T> types;
        std::vector<size_t> reps;
    };
}

#endif //CONSTRAINT_DATA_H
