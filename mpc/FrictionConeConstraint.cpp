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
            ad::DerivativeOrder::FirstOrder, 3, 5,
            compile_derivs);
    }

    int FrictionConeConstraint::GetNumConstraints() const {
        return constraint_function_->GetRangeSize();
    }


    std::pair<matrixx_t, vectorx_t> FrictionConeConstraint::GetLinearization(const vectorx_t &f_lin) const {
        matrixx_t jac;

        vectorx_t x_zero = vectorx_t::Zero(constraint_function_->GetDomainSize());
        vectorx_t p(constraint_function_->GetParameterSize());
        p << f_lin, friction_margin_, friction_coef_;

        constraint_function_->GetJacobian(x_zero, p, jac);

        vectorx_t y;
        constraint_function_->GetFunctionValue(x_zero, p, y);

        // matrixx_t A = matrixx_t::Zero(constraint_function_->GetRangeSize(), constraint_function_->GetDomainSize());
        // A.topRows(constraint_function_->GetRangeSize()) = jac;
        // A(constraint_function_->GetRangeSize(), 0) = 1;
        // A(constraint_function_->GetRangeSize() + 1, 1) = 1;
        // A(constraint_function_->GetRangeSize() + 2, 2) = 1;

        // vectorx_t lin = vectorx_t::Zero(constraint_function_->GetRangeSize());
        // lin.head(constraint_function_->GetRangeSize()) = y;
        // lin.tail<3>() = f_lin;

        return {jac, y}; //lin};
    }

    vectorx_t FrictionConeConstraint::GetViolation(const vectorx_t &F, const vectorx_t& dF) {
        vectorx_t x(constraint_function_->GetDomainSize());
        x << dF;
        vectorx_t p(constraint_function_->GetParameterSize());
        p << F, friction_margin_, friction_coef_;

        vectorx_t fcn_vio;
        constraint_function_->GetFunctionValue(x, p, fcn_vio);

        vectorx_t violation(constraint_function_->GetRangeSize());
        violation.head(constraint_function_->GetRangeSize()) = fcn_vio;
        // violation.tail<3>() = dF + F;

        return violation;
    }

    void FrictionConeConstraint::ConeConstraint(const ad::ad_vector_t& df, const ad::ad_vector_t& fk_margin_coef, ad::ad_vector_t& violation) const {
        const ad::ad_vector_t f = df + fk_margin_coef.head<CONTACT_3DOF>();
        const ad::adcg_t& margin = fk_margin_coef(3);
        const ad::adcg_t& friction_coef = fk_margin_coef(4);

        // Linear cone, >= 0
        violation.resize(4);
        violation(0) = friction_coef * f(2) - f(0);
        violation(1) = friction_coef * f(2) - f(1);
        violation(2) = friction_coef * f(2) + f(0);
        violation(3) = friction_coef * f(2) + f(1);
        // violation(4) = f(2);

        // Nonlinear
        // violation.resize(1);
        // violation(0) = friction_coef*f(2) - CppAD::sqrt(f(0)*f(0) + f(1)*f(1) + margin*margin);
    }

}