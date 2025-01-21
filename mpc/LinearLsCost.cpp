//
// Created by zolkin on 1/20/25.
//

#include "LinearLsCost.h"

namespace torc::mpc {
    LinearLsCost::LinearLsCost(int first_node, int last_node, const std::string &name, const vectorx_t &weights,
        const std::filesystem::path& deriv_lib_path, bool compile_derivs)
    : Cost(first_node, last_node, name, weights), var_size_(weights.size()) {
        cost_function_ = std::make_unique<ad::CppADInterface> (
                        std::bind(&LinearLsCost::CostFunction, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                        name_ + "_linear_ls_cost",
                        deriv_lib_path,
                        torc::ad::DerivativeOrder::SecondOrder, weights.size(), 3*weights.size(),
                        compile_derivs);
        std::cout << "weights size: " << weights_.size() << ", weights:\n" << weights_.transpose() << std::endl;
    }

    std::pair<matrixx_t, vectorx_t> LinearLsCost::GetQuadraticApprox(const vectorx_t &x_lin, const vectorx_t& p) {
        if (x_lin.size() != var_size_ || p.size() != var_size_) {
            std::cerr << "x lin size: " << x_lin.size() << ", p size: " << p.size() << std::endl;
            std::cerr << "expected " << var_size_ << std::endl;
            throw std::runtime_error("[LinearLSCost] x_lin or p has the wrong size!");
        }

        vectorx_t x_zero = vectorx_t::Zero(cost_function_->GetDomainSize());

        matrixx_t hessian;
        vectorx_t linear_term;

        vectorx_t p_ad(cost_function_->GetParameterSize());
        p_ad << x_lin, p, weights_; // TODO: Check to make sure this is filled in by here

        matrixx_t jac;
        cost_function_->GetJacobian(x_zero, p_ad, jac);

        vectorx_t y;
        cost_function_->GetFunctionValue(x_zero, p_ad, y);
        linear_term = 2*jac.transpose()*y;

        cost_function_->GetHessian(x_zero, p_ad, 2*y, hessian);
        hessian += 2*jac.transpose() * jac;

        return {hessian, linear_term};
    }

    void LinearLsCost::CostFunction(const torc::ad::ad_vector_t& dx,
                            const torc::ad::ad_vector_t& xref_xtarget_weight,
                            torc::ad::ad_vector_t& x_diff) const {
        // Get the current velocity
        x_diff = dx + xref_xtarget_weight.head(var_size_);

        // Get the difference between the velocity and its target
        x_diff = x_diff - xref_xtarget_weight.segment(var_size_, var_size_);

        // Multiply by the weights
        for (int i = 0; i < var_size_; i++) {
            x_diff(i) = x_diff(i) * xref_xtarget_weight(2*var_size_ + i);
        }
    }

}