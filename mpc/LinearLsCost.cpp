//
// Created by zolkin on 1/20/25.
//

#include "LinearLsCost.h"

namespace torc::mpc {
    LinearLsCost::LinearLsCost(int first_node, int last_node, const std::string &name,
        const std::filesystem::path& deriv_lib_path, bool compile_derivs, int var_size)
    : Cost(first_node, last_node, name) {
        cost_function_ = std::make_unique<ad::CppADInterface> (
                        std::bind(&LinearLsCost::CostFunction, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                        name_ + "_linear_ls_cost",
                        deriv_lib_path,
                        torc::ad::DerivativeOrder::SecondOrder, var_size, 3*var_size,
                        compile_derivs);
    }

    std::pair<matrixx_t, vectorx_t> LinearLsCost::GetQuadraticApprox(const vectorx_t &x_lin, const vectorx_t& p,
        const vectorx_t& weight) {
        if (x_lin.size() != weight.size() || p.size() != weight.size()) {
            std::cerr << "x lin size: " << x_lin.size() << ", p size: " << p.size() << std::endl;
            std::cerr << "expected " << weight.size() << std::endl;
            throw std::runtime_error("[LinearLSCost] x_lin or p has the wrong size!");
        }

        vectorx_t x_zero = vectorx_t::Zero(cost_function_->GetDomainSize());

        matrixx_t hessian;
        vectorx_t linear_term;

        vectorx_t p_ad(cost_function_->GetParameterSize());
        p_ad << x_lin, p, weight;

        matrixx_t jac;
        cost_function_->GetJacobian(x_zero, p_ad, jac);

        vectorx_t y;
        cost_function_->GetFunctionValue(x_zero, p_ad, y);
        linear_term = 2*jac.transpose()*y;

        cost_function_->GetHessian(x_zero, p_ad, 2*y, hessian);
        hessian += 2*jac.transpose() * jac;

        return {hessian, linear_term};
    }

    double LinearLsCost::GetCost(const vectorx_t &x, const vectorx_t &dx, const vectorx_t &p, const vectorx_t& weight) {
        vectorx_t x_ad(cost_function_->GetDomainSize());
        x_ad << dx;

        vectorx_t y;

        vectorx_t p_ad(cost_function_->GetParameterSize());
        p_ad << x, p, weight;

        cost_function_->GetFunctionValue(x_ad, p_ad, y);
        return y.norm();
    }


    void LinearLsCost::CostFunction(const torc::ad::ad_vector_t& dx,
                            const torc::ad::ad_vector_t& xref_xtarget_weight,
                            torc::ad::ad_vector_t& x_diff) const {
        // Get the current velocity
        x_diff = dx + xref_xtarget_weight.head(dx.size());

        // Get the difference between the velocity and its target
        x_diff = x_diff - xref_xtarget_weight.segment(dx.size(), dx.size());

        // Multiply by the weights
        for (int i = 0; i < dx.size(); i++) {
            x_diff(i) = x_diff(i) * xref_xtarget_weight(2*dx.size() + i);
        }
    }

}