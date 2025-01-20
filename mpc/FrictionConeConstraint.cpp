//
// Created by zolkin on 1/18/25.
//

#include "FrictionConeConstraint.h"

namespace torc::mpc {
    FrictionConeConstraint::FrictionConeConstraint(int first_node, int last_node, const std::string& name,
        double friction_coef, double friction_margin, const fs::path& deriv_lib_path,
        bool compile_derivs)
        : Constraint(first_node, last_node, name), friction_coef_(friction_coef), friction_margin_(friction_margin),
            constraint_function_(
            std::bind(&FrictionConeConstraint::ConeConstraint, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
            name_ + "_friction_cone_constraint",
            deriv_lib_path,
            ad::DerivativeOrder::FirstOrder, 3, 4,
            compile_derivs) {
    }

    int FrictionConeConstraint::GetNumConstraints() const {
        return constraint_function_.GetRangeSize();
    }


    std::pair<matrixx_t, vectorx_t> FrictionConeConstraint::GetLinearization(const vectorx_t &f_lin) const {
        matrixx_t jac;

        vectorx_t x_zero = vectorx_t::Zero(constraint_function_.GetDomainSize());
        vectorx_t p(constraint_function_.GetParameterSize());
        p << f_lin, friction_margin_;

        constraint_function_.GetJacobian(x_zero, p, jac);

        vectorx_t y;
        constraint_function_.GetFunctionValue(x_zero, p, y);

        return {jac, y};
    }


    void FrictionConeConstraint::ConeConstraint(const ad::ad_vector_t& df, const ad::ad_vector_t& fk_margin, ad::ad_vector_t& violation) const {
        const ad::ad_vector_t f = df + fk_margin.head<CONTACT_3DOF>();
        const ad::adcg_t& margin = fk_margin(3);

        violation.resize(1);
        violation(0) = friction_coef_*f(2) - CppAD::sqrt(f(0)*f(0) + f(1)*f(1) + margin*margin);
    }

}