//
// Created by zolkin on 8/21/24.
//

#ifndef AUTO_DIFF_TYPES_H
#define AUTO_DIFF_TYPES_H

#include <Eigen/Core>

// TODO: What is finding this in cmake?
// Somehow all of the packages in usr/local/lib are automatically being added
#include <cppad/cg.hpp>

namespace torc::ad {
    namespace ADCG = CppAD::cg;
    namespace AD = CppAD;

    using vectorx_t = Eigen::VectorX<double>;
    using matrixx_t = Eigen::MatrixX<double>;
    using cg_t = ADCG::CG<double>;
    using adcg_t = CppAD::AD<cg_t>;

    using sparsity_pattern_t = std::vector<std::set<size_t>>;
    using ad_vector_t = Eigen::VectorX<adcg_t>;
    using ad_matrix_t = Eigen::MatrixX<adcg_t>;

    using ad_fcn_t = std::function<void(const ad_vector_t&, const ad_vector_t&, ad_vector_t&)>;
}   // namepsace torc::ad

#endif //AUTO_DIFF_TYPES_H
