//
// Created by zolkin on 1/18/25.
//

#include "FrictionConeConstraint.h"

namespace torc::mpc {
    FrictionConeConstraint::FrictionConeConstraint(int first_node, int last_node, const std::string& name,
        double friction_coef, double friction_margin, const fs::path& deriv_lib_path,
        bool compile_derivs)
        : Constraint(first_node, last_node, name), friction_coef_(friction_coef), friction_margin_(friction_margin) {
        constraint_function_ = std::make_unique<ad::CppADInterface>(
            std::bind(&FrictionConeConstraint::ConeConstraint, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
            name_ + "_friction_cone_constraint",
            deriv_lib_path,
            ad::DerivativeOrder::FirstOrder, 3, 4,
            compile_derivs);

        if (constraint_function_->GetRangeSize() != 1) {
            throw std::runtime_error("Friction cone should have a range size of 1");
        }

    }

    int FrictionConeConstraint::GetNumConstraints() const {
        return constraint_function_->GetRangeSize() + CONTACT_3DOF; // Cone and 0 in the air
    }


    std::pair<matrixx_t, vectorx_t> FrictionConeConstraint::GetLinearization(const vectorx_t &f_lin) const {
        matrixx_t jac;

        vectorx_t x_zero = vectorx_t::Zero(constraint_function_->GetDomainSize());
        vectorx_t p(constraint_function_->GetParameterSize());
        p << f_lin, friction_margin_;

        constraint_function_->GetJacobian(x_zero, p, jac);

        vectorx_t y;
        constraint_function_->GetFunctionValue(x_zero, p, y);

        matrixx_t A = matrixx_t::Zero(constraint_function_->GetRangeSize() + 3, constraint_function_->GetDomainSize());
        A.row(0) = jac;
        A(1, 0) = 1;
        A(2, 1) = 1;
        A(3, 2) = 1;

        vectorx_t lin = vectorx_t::Zero(constraint_function_->GetRangeSize() + 3);
        lin(0) = y(0);
        lin.tail<3>() = f_lin;

        return {A, lin};
    }

    vectorx_t FrictionConeConstraint::GetViolation(const vectorx_t &F, double margin) {
        vectorx_t x_zero = vectorx_t::Zero(constraint_function_->GetDomainSize());
        vectorx_t p(constraint_function_->GetParameterSize());
        p << F, margin;

        vectorx_t fcn_vio;
        constraint_function_->GetFunctionValue(x_zero, p, fcn_vio);

        vectorx_t violation(4);
        violation(0) = fcn_vio(0);
        violation.tail<3>() = F;

        return violation;
    }

    void FrictionConeConstraint::ConeConstraint(const ad::ad_vector_t& df, const ad::ad_vector_t& fk_margin, ad::ad_vector_t& violation) const {
        const ad::ad_vector_t f = df + fk_margin.head<CONTACT_3DOF>();
        const ad::adcg_t& margin = fk_margin(3);

        violation.resize(1);
        violation(0) = friction_coef_*f(2) - CppAD::sqrt(f(0)*f(0) + f(1)*f(1) + margin*margin);
    }

}