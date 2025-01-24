//
// Created by zolkin on 1/20/25.
//

#include "ConfigTrackingCost.h"

#include "pinocchio_interface.h"

namespace torc::mpc {
    ConfigTrackingCost::ConfigTrackingCost(int start_node, int last_node, const std::string& name, const vectorx_t &weights,
        const std::filesystem::path &deriv_lib_path, bool compile_derivs, const models::FullOrderRigidBody& model)
    : Cost(start_node, last_node, name, weights), nq_(model.GetConfigDim()), nv_(model.GetVelDim()), var_size_(weights.size()) {
        cost_function_ = std::make_unique<ad::CppADInterface>(
            std::bind(&ConfigTrackingCost::CostFunction, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
            name_ + "_nonlinear_ls_cost",
            deriv_lib_path,
            torc::ad::DerivativeOrder::FirstOrder, model.GetVelDim(), 2*model.GetConfigDim() + weights.size(),
            compile_derivs);
    }

    std::pair<matrixx_t, vectorx_t> ConfigTrackingCost::GetQuadraticApprox(const vectorx_t &x_lin, const vectorx_t &p) {
        if (x_lin.size() != nq_ || p.size() != nq_) {
            throw std::runtime_error("[Cost Function] configuration approx reference or target has the wrong size!");
        }

        vectorx_t ad_p(cost_function_->GetParameterSize());
        ad_p << x_lin, p, weights_;

        matrixx_t hessian_term;
        vectorx_t linear_term;

        matrixx_t jac;
        cost_function_->GetGaussNewton(vectorx_t::Zero(nv_), ad_p, jac, hessian_term);
        hessian_term = 2*hessian_term;

        vectorx_t y;
        cost_function_->GetFunctionValue(vectorx_t::Zero(nv_), ad_p, y);
        linear_term = 2*jac.transpose()*y;

        return {hessian_term, linear_term};
    }

    double ConfigTrackingCost::GetCost(const vectorx_t &x, const vectorx_t &dx, const vectorx_t &p) const {
        vectorx_t x_ad(cost_function_->GetDomainSize());
        x_ad << dx;

        vectorx_t y;

        vectorx_t p_ad(cost_function_->GetParameterSize());
        p_ad << x, p, weights_;

        cost_function_->GetFunctionValue(x_ad, p_ad, y);
        return y.squaredNorm();
    }


    void ConfigTrackingCost::CostFunction(const torc::ad::ad_vector_t &dx,
        const torc::ad::ad_vector_t &xref_xtarget_weight, torc::ad::ad_vector_t &x_diff) const {
        // I'd like to just call pinocchio's integrate here, but that will require a templated model, which I currently don't have

        const ad::ad_vector_t& q_lin = xref_xtarget_weight.head(nq_);
        const ad::ad_vector_t& q_target = xref_xtarget_weight.segment(nq_, nq_);

        ad::ad_vector_t q = models::ConvertdqToq(dx, q_lin);

        x_diff.resize(nv_);

        x_diff.head<3>() = q.head<3>() - q_target.head<3>();

        // Floating base orientation difference
        Eigen::Quaternion<torc::ad::adcg_t> q_quat, quat_target;
        q_quat.coeffs() = q.segment<4>(3);
        quat_target.coeffs() = q_target.segment<4>(3);
        // Eigen's inverse has an if statement, so we can't use it in codegen
        quat_target = Eigen::Quaternion<torc::ad::adcg_t>(quat_target.conjugate().coeffs() / quat_target.squaredNorm());   // Assumes norm > 0
        x_diff.segment<3>(3) = pinocchio::quaternion::log3(quat_target * q_quat);    // TODO: Double check

        x_diff.tail(nv_ - 6) = q.tail(nq_ - 7) - q_target.tail(nq_ - 7);

        for (int i = 0; i < nv_; i++) {
            x_diff(i) = x_diff(i) * xref_xtarget_weight(2*nq_ + i);
        }
    }


}
