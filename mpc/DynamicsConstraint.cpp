//
// Created by zolkin on 1/18/25.
//

#include "pinocchio_model.h"
#include "DynamicsConstraint.h"

#include <pinocchio/algorithm/joint-configuration.hpp>

#include "pinocchio_interface.h"



namespace torc::mpc {
    DynamicsConstraint::DynamicsConstraint(const models::FullOrderRigidBody& model,
        const std::vector<std::string>& contact_frames, const std::string& name, const fs::path& deriv_lib_path,
        bool compile_derivs,
        int first_node, int last_node)
        : Constraint(first_node, last_node, name), model_(model), pin_data_(model.GetModel()) {

        vel_dim_ = model_.GetVelDim();
        config_dim_ = model_.GetConfigDim();
        tau_dim_ = model_.GetVelDim() - FLOATING_VEL;
        num_contacts_ = contact_frames.size();
        contact_frames_ = contact_frames;

        // Make the auto diff function
        dynamics_function_ = std::make_unique<ad::CppADInterface>(
            std::bind(&DynamicsConstraint::InverseDynamics, this, contact_frames, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3),
            name_ + "_dynamics_constraint",
            deriv_lib_path,
            ad::DerivativeOrder::FirstOrder, 3*vel_dim_ + tau_dim_ + CONTACT_3DOF*num_contacts_,
            1 + config_dim_ + 2*vel_dim_ + tau_dim_ + CONTACT_3DOF*num_contacts_,
            compile_derivs
        );

        integration_function_ = std::make_unique<ad::CppADInterface>(
            std::bind(&DynamicsConstraint::IntegrationConstraint, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
            name_ + "_integration_constraint",
            deriv_lib_path,
            ad::DerivativeOrder::FirstOrder, 3*vel_dim_, 1 + 2*config_dim_ + vel_dim_,
            compile_derivs
        );
    }

    void DynamicsConstraint::GetLinDynamics(const vectorx_t &q1_lin, const vectorx_t &q2_lin,
        const vectorx_t &v1_lin, const vectorx_t& v2_lin, const vectorx_t &tau_lin, const vectorx_t &force_lin,
        double dt, bool boundary, matrixx_t& A, matrixx_t& B, vectorx_t& b) {
        // --------- Dynamics --------- //
        vectorx_t x_zero = vectorx_t::Zero(dynamics_function_->GetDomainSize());
        vectorx_t p(dynamics_function_->GetParameterSize());
        p << q1_lin, v1_lin, v2_lin, tau_lin, force_lin, dt;

        matrixx_t dyn_jac;
        dynamics_function_->GetJacobian(x_zero, p, dyn_jac);

        vectorx_t fbar;
        dynamics_function_->GetFunctionValue(x_zero, p, fbar);

        // --------- Integration --------- //
        x_zero.resize(integration_function_->GetDomainSize());
        x_zero.setZero();

        p.resize(integration_function_->GetParameterSize());
        p << dt, q1_lin, q2_lin, v1_lin;

        matrixx_t int_jac;
        integration_function_->GetJacobian(x_zero, p, int_jac);

        vectorx_t int_fbar;
        integration_function_->GetFunctionValue(x_zero, p, int_fbar);

        if (boundary) {
            A.setZero();
            B.setZero();
            b.setZero();

            matrixx_t dv2_inv = dyn_jac.middleCols(2*vel_dim_, vel_dim_).inverse();
            matrixx_t dq2_inv = int_jac.middleCols(vel_dim_, vel_dim_).inverse();

            // Integration
            A.topRows(vel_dim_) << -dq2_inv*int_jac.leftCols(vel_dim_), -dq2_inv*int_jac.middleCols(2*vel_dim_, vel_dim_);

            // Dynamics
            matrixx_t A_dyn_temp = -dv2_inv*dyn_jac.leftCols(2*vel_dim_);
            A.bottomRows<FLOATING_VEL>() = A_dyn_temp.topRows<FLOATING_VEL>();

            matrixx_t B_dyn_temp = matrixx_t::Zero(vel_dim_, tau_dim_ + CONTACT_3DOF*num_contacts_);
            B_dyn_temp << -dv2_inv*dyn_jac.middleCols(3*vel_dim_, tau_dim_),
                -dv2_inv*dyn_jac.middleCols(3*vel_dim_ + tau_dim_, CONTACT_3DOF*num_contacts_);
            B.bottomRows<FLOATING_VEL>() = B_dyn_temp.topRows<FLOATING_VEL>();

            b << -dq2_inv*int_fbar, -(dv2_inv*fbar).topRows<FLOATING_VEL>();
        } else {
            A.setZero();
            B.setZero();
            b.setZero();

            matrixx_t dv2_inv = dyn_jac.middleCols(2*vel_dim_, vel_dim_).inverse();
            matrixx_t dq2_inv = int_jac.middleCols(vel_dim_, vel_dim_).inverse();

            // Integration constraint
            A.topRows(vel_dim_) << -dq2_inv*int_jac.leftCols(vel_dim_), -dq2_inv*int_jac.middleCols(2*vel_dim_, vel_dim_);
            B.topRows(vel_dim_).setZero();

            // Dynamics constraint
            A.bottomRows(vel_dim_) = -dv2_inv*dyn_jac.leftCols(2*vel_dim_);
            B.bottomRows(vel_dim_) << dyn_jac.middleCols(3*vel_dim_, tau_dim_),
                dyn_jac.middleCols(3*vel_dim_ + tau_dim_, CONTACT_3DOF*num_contacts_);
            B.bottomRows(vel_dim_) = -dv2_inv*B.bottomRows(vel_dim_);

            b << -dq2_inv*int_fbar, -dv2_inv*fbar;
        }
    }

    void DynamicsConstraint::InverseDynamics(const std::vector<std::string> &frames,
        const ad::ad_vector_t &dqk_dvk_dvkp1_dtauk_dfk, const ad::ad_vector_t &qk_vk_vkp1_tauk_fk_dt,
        ad::ad_vector_t &violation) {

        // Decision variables
        const ad::ad_vector_t& dqk = dqk_dvk_dvkp1_dtauk_dfk.head(vel_dim_);
        const ad::ad_vector_t& dvk = dqk_dvk_dvkp1_dtauk_dfk.segment(vel_dim_, vel_dim_);
        const ad::ad_vector_t& dvkp1 = dqk_dvk_dvkp1_dtauk_dfk.segment(2*vel_dim_, vel_dim_);
        ad::ad_vector_t dtauk(vel_dim_);
        dtauk << ad::ad_vector_t::Zero(FLOATING_VEL), dqk_dvk_dvkp1_dtauk_dfk.segment(3*vel_dim_, tau_dim_);
        const ad::ad_vector_t& dfk = dqk_dvk_dvkp1_dtauk_dfk.segment(3*vel_dim_ + tau_dim_, CONTACT_3DOF*num_contacts_);

        // Reference trajectory
        const ad::ad_vector_t& qk = qk_vk_vkp1_tauk_fk_dt.head(config_dim_);
        const ad::ad_vector_t& vk = qk_vk_vkp1_tauk_fk_dt.segment(config_dim_, vel_dim_);
        const ad::ad_vector_t& vkp1 = qk_vk_vkp1_tauk_fk_dt.segment(config_dim_ + vel_dim_, vel_dim_);
        ad::ad_vector_t tauk(vel_dim_);
        tauk << ad::ad_vector_t::Zero(FLOATING_VEL), qk_vk_vkp1_tauk_fk_dt.segment(config_dim_ + 2*vel_dim_, tau_dim_);
        const ad::ad_vector_t& fk = qk_vk_vkp1_tauk_fk_dt.segment(config_dim_ + 2*vel_dim_ + tau_dim_, CONTACT_3DOF*num_contacts_);
        const ad::adcg_t& dt = qk_vk_vkp1_tauk_fk_dt(config_dim_ + 2*vel_dim_ + tau_dim_ + CONTACT_3DOF*num_contacts_);

        // Current values
        const ad::ad_vector_t qk_curr = models::ConvertdqToq(dqk, qk);
        const ad::ad_vector_t vk_curr = dvk + vk;
        const ad::ad_vector_t vkp1_curr = dvkp1 + vkp1;
        const ad::ad_vector_t tauk_curr = dtauk + tauk;
        const ad::ad_vector_t fk_curr = dfk + fk;

        // Intermediate values
        const ad::ad_vector_t a = (vkp1_curr - vk_curr)/dt; // TODO: Do I need to account for the different local frames somehow?
        std::vector<models::ExternalForce<ad::adcg_t>> f_ext;

        int idx = 0;
        for (const auto& frame : frames) {
            f_ext.emplace_back(frame, fk_curr.segment<CONTACT_3DOF>(idx));
            idx += CONTACT_3DOF;
        }

        // Compute error
        const ad::ad_vector_t tau_id = models::InverseDynamics(model_.GetADPinModel(), *model_.GetADPinData(),
            qk_curr, vk_curr, a, f_ext);

        violation = tau_id - tauk_curr;
    }

    int DynamicsConstraint::GetNumConstraints() const {
        return dynamics_function_->GetRangeSize() + integration_function_->GetRangeSize();
    }

    void DynamicsConstraint::IntegrationConstraint(const ad::ad_vector_t& dqk_dqkp1_dvk,
         const ad::ad_vector_t& dt_qkbar_qkp1bar_vk, ad::ad_vector_t& violation) {
        // From the reference trajectory
        const ad::adcg_t& dt = dt_qkbar_qkp1bar_vk(0);
        const ad::ad_vector_t& qkbar = dt_qkbar_qkp1bar_vk.segment(1, config_dim_);
        const ad::ad_vector_t& qkp1bar = dt_qkbar_qkp1bar_vk.segment(1 + config_dim_, config_dim_);
        const ad::ad_vector_t& vkbar = dt_qkbar_qkp1bar_vk.segment(1 + 2*config_dim_, vel_dim_);

        // Changes from decision variables
        const ad::ad_vector_t& dqk = dqk_dqkp1_dvk.head(vel_dim_);
        const ad::ad_vector_t& dqkp1 = dqk_dqkp1_dvk.segment(vel_dim_, vel_dim_);
        const ad::ad_vector_t& dvk = dqk_dqkp1_dvk.segment(2*vel_dim_, vel_dim_);

        // Get the current configuration
        const ad::ad_vector_t qk = torc::models::ConvertdqToq(dqk, qkbar);
        const ad::ad_vector_t qkp1 = torc::models::ConvertdqToq(dqkp1, qkp1bar);

        // Velocity
        const ad::ad_vector_t vk = vkbar + dvk;

        const ad::ad_vector_t v = dt*vk;

        const ad::ad_vector_t qkp1_new = pinocchio::integrate(model_.GetADPinModel(), qk, v);

        // Floating base position differences
        violation.resize(vel_dim_);
        violation.head<POS_VARS>() = qkp1_new.head<POS_VARS>() - qkp1.head<POS_VARS>();

        // Quaternion difference in the tangent space
        Eigen::Quaternion<ad::adcg_t> quat_kp1(qkp1.segment<QUAT_VARS>(POS_VARS));
        Eigen::Quaternion<ad::adcg_t> quat_kp1_new(qkp1_new.segment<QUAT_VARS>(POS_VARS));

        // Eigen's inverse has an if statement, so we can't use it in codegen
        quat_kp1 = Eigen::Quaternion<torc::ad::adcg_t>(quat_kp1.conjugate().coeffs() / quat_kp1.squaredNorm());   // Assumes norm > 0
        violation.segment<3>(POS_VARS) = pinocchio::quaternion::log3(quat_kp1 * quat_kp1_new);

        // Joint differences
        violation.tail(vel_dim_ - FLOATING_VEL) = qkp1_new.tail(config_dim_ - FLOATING_BASE) - qkp1.tail(config_dim_ - FLOATING_BASE);
    }

    std::pair<vectorx_t, vectorx_t> DynamicsConstraint::GetViolation(const vectorx_t &q1_lin, const vectorx_t &q2_lin,
    const vectorx_t &v1_lin, const vectorx_t &v2_lin, const vectorx_t &tau_lin,
    const vectorx_t &force_lin, double dt, const vectorx_t& dq1, const vectorx_t& dq2,
    const vectorx_t& dv1, const vectorx_t& dv2, const vectorx_t& dtau, const vectorx_t& dforce) {
        vectorx_t tau_casadi = vectorx_t::Zero(vel_dim_);
        tau_casadi.tail(tau_dim_) = tau_lin;

        vectorx_t dtau_full = vectorx_t::Zero(vel_dim_);
        dtau_full.tail(tau_dim_) = dtau;


        // Check against C++
        vectorx_t q_eval = models::ConvertdqToq(dq1, q1_lin);
        vectorx_t v_eval = (v1_lin + dv1);
        vectorx_t tau_eval = (tau_casadi + dtau_full);
        vectorx_t force_eval = force_lin + dforce;

        int idx = 0;
        std::vector<models::ExternalForce<double>> f_ext;
        for (const auto& frame : contact_frames_) {
            f_ext.emplace_back(frame, force_eval.segment<CONTACT_3DOF>(idx));
            idx += CONTACT_3DOF;
        }

        vectorx_t a_pin = models::ForwardDynamics(model_.GetModel(), pin_data_, q_eval, v_eval, tau_eval, f_ext);
        vectorx_t check_dv2 = dt*a_pin + v_eval - v2_lin;

        vectorx_t dyn_vio = dv2 - check_dv2;


        vectorx_t int_violation(integration_function_->GetRangeSize());
        vectorx_t x(integration_function_->GetDomainSize());
        x << dq1, dq2, dv1;
        vectorx_t p(integration_function_->GetParameterSize());
        p << dt, q1_lin, q2_lin, v1_lin;
        integration_function_->GetFunctionValue(x, p, int_violation);

        return {int_violation, dyn_vio};
    }
}
