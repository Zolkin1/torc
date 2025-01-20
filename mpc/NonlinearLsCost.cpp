//
// Created by zolkin on 1/20/25.
//

#include "NonlinearLsCost.h"

namespace torc::mpc {
    NonlinearLsCost::NonlinearLsCost(int first_node, int last_node, const std::string &name, const vectorx_t &weights,
        const std::filesystem::path& deriv_lib_path, bool compile_derivs)
    : Cost(first_node, last_node, name, weights), cost_function_(
                    std::bind(&NonlinearLsCost::CostFunction, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                    name_ + "_nonlinear_ls_cost",
                    deriv_lib_path,
                    torc::ad::DerivativeOrder::SecondOrder, weights.size(), 3*weights.size(),
                    compile_derivs), var_size_(weights.size()) {}

}