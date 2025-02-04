//
// Created by zolkin on 2/3/25.
//

#include "pinocchio_interface.h"
#include "wbc_controller.h"

#include <eigen_utils.h>
#include <torc_timer.h>
#include <pinocchio/algorithm/frames.hpp>

namespace torc::controller {
    WbcController::WbcController(const models::FullOrderRigidBody &model,
        const std::vector<std::string> &contact_frames, const vectorx_t &base_weight,
        const vectorx_t &joint_weight, const vectorx_t &tau_weight, const vectorx_t &force_weight,
        const vectorx_t &kp, const vectorx_t &kd, double friction_coef, bool verbose,
        const std::filesystem::path deriv_lib_path, bool compile_derivs)
            : model_(model), contact_frames_(contact_frames), verbose_(verbose), pin_data_(model.GetModel()),
                mu_(friction_coef), kp_(kp), kd_(kd), base_weight_(base_weight), joint_weight_(joint_weight),
                tau_weight_(tau_weight), force_weight_(force_weight) {

        nv_ = model_.GetVelDim();
        ntau_ = nv_ - FLOATING_VEL;
        ncontact_frames_ = contact_frames_.size();
        nF_ = CONTACT_3DOF*ncontact_frames_;
        nd_ = nv_ + ntau_ + nF_;

        // Make the auto diff function
        inverse_dynamics_ = std::make_unique<ad::CppADInterface>(
            std::bind(&WbcController::DynamicsFunction, this, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3),
            "wbc_dynamics_constraint",
            deriv_lib_path,
            ad::DerivativeOrder::FirstOrder, nd_, model_.GetConfigDim() + nv_,
            compile_derivs
        );
    }

    vectorx_t WbcController::ComputeControl(const vectorx_t &q, const vectorx_t &v,
        const vectorx_t &q_des, const vectorx_t &v_des, const vectorx_t &tau_des, const vectorx_t &F_des,
        std::vector<bool> in_contact) {
        torc::utils::TORCTimer timer;

        timer.Tic();
        if (in_contact.size() != ncontact_frames_) {
            throw std::runtime_error("[WBC] contact vector does not match contact frames size!");
        }

        // For now, updating the size with every solve depending on the contacts
        ncontacts_ = 0;
        for (const auto& contact : in_contact) {
            if (contact) {
                ncontacts_++;
            }
        }

        // Dynamics, no-slip, friction cone, no force, torque box
        nc_ = nv_ + ncontacts_*3 + ncontacts_*4 + (ncontact_frames_ - ncontacts_)*3 + nv_;

        // --------- Constraints ---------- //
        matrixx_t A = matrixx_t::Zero(nc_, nd_);
        osqp_instance_.lower_bounds = vectorx_t::Zero(nc_);
        osqp_instance_.upper_bounds = vectorx_t::Zero(nc_);

        int rowc = 0;
        vectorx_t lb(nv_), ub(nv_);
        A.topRows(nv_) = DynamicsConstraint(q, v, lb, ub);
        osqp_instance_.lower_bounds.segment(rowc, nv_) = lb;
        osqp_instance_.upper_bounds.segment(rowc, nv_) = ub;
        rowc += nv_;

        lb.resize(4*ncontacts_ + (ncontact_frames_ - ncontacts_)*3);
        ub.resize(4*ncontacts_ + (ncontact_frames_ - ncontacts_)*3);
        A.middleRows(rowc, 4*ncontacts_ + (ncontact_frames_ - ncontacts_)*3) = ForceConstraint(in_contact, lb, ub);
        osqp_instance_.lower_bounds.segment(rowc, 4*ncontacts_ + (ncontact_frames_ - ncontacts_)*3) = lb;
        osqp_instance_.upper_bounds.segment(rowc, 4*ncontacts_ + (ncontact_frames_ - ncontacts_)*3) = ub;
        rowc += 4*ncontacts_ + (ncontact_frames_ - ncontacts_)*3;

        // TODO: Fix!
        // lb.resize(ncontacts_*3);
        // ub.resize(ncontacts_*3);
        // A.middleRows(rowc, ncontacts_*3) = HolonomicConstraint(q, v, in_contact, lb, ub);
        // osqp_instance_.lower_bounds.segment(rowc, ncontacts_*3) = lb;
        // osqp_instance_.upper_bounds.segment(rowc, ncontacts_*3) = ub;
        rowc += 3*ncontacts_;

        A.middleRows(rowc, ntau_) = TorqueBoxConstraint();
        osqp_instance_.lower_bounds.segment(rowc, ntau_) = -model_.GetTorqueJointLimits().tail(ntau_);
        osqp_instance_.upper_bounds.segment(rowc, ntau_) = model_.GetTorqueJointLimits().tail(ntau_);

        // --------- Cost ---------- //
        matrixx_t P = matrixx_t::Zero(nd_, nd_);
        osqp_instance_.objective_vector = vectorx_t::Zero(nd_);

        const auto [stateP, stateq] = StateTracking(q, v, q_des, v_des);
        P.topLeftCorner(nv_, nv_) = stateP;
        osqp_instance_.objective_vector.head(nv_) = stateq;

        const auto [tauP, tauq] = TorqueTracking(tau_des);
        P.block(nv_, nv_, ntau_, ntau_) = tauP;
        osqp_instance_.objective_vector.segment(nv_, ntau_) = tauq;

        const auto [FP, Fq] = ForceTracking(F_des);
        P.bottomRightCorner(CONTACT_3DOF*ncontact_frames_, CONTACT_3DOF*ncontact_frames_) = FP;
        osqp_instance_.objective_vector.segment(nv_ + ntau_, nF_) = Fq;

        // --------- Solve ---------- //
        // TODO: Consider doing this once in the constructor if its slow
        // Initialize OSQP
        osqp_instance_.objective_matrix = P.sparseView();
        osqp_instance_.constraint_matrix = A.sparseView();

        // std::cerr << "P:\n" << P << std::endl;
        // std::cerr << "A:\n" << A << std::endl;

        osqp_settings_.verbose = verbose_;
        auto status = osqp_solver_.Init(osqp_instance_, osqp_settings_);    // Takes about 5ms for 20 nodes
        if (!status.ok()) {
            throw std::runtime_error("[WBC] Could not initialize OSQP!");
        }
        timer.Toc();

        osqp::OsqpExitCode exit_code = osqp_solver_.Solve();
        if (exit_code != osqp::OsqpExitCode::kOptimal) {
            throw std::runtime_error("[WBC] Could not solve QP!");
        }

        if (verbose_) {
            std::cout << "tau: " << osqp_solver_.primal_solution().segment(nv_, ntau_).transpose() << std::endl;
            std::cout << "a: " << osqp_solver_.primal_solution().head(nv_).transpose() << std::endl;
            std::cout << "F: " << osqp_solver_.primal_solution().segment(nv_ + ntau_, nF_).transpose() << std::endl;
            std::cout << "Setup took " << timer.Duration<std::chrono::microseconds>().count()*1e-3 << " ms" << std::endl;
        }

        // Return the torques
        return osqp_solver_.primal_solution().segment(nv_, ntau_);
    }

    matrixx_t WbcController::HolonomicConstraint(const vectorx_t &q, const vectorx_t &v,
        const std::vector<bool> &in_contact, vectorx_t& lb, vectorx_t& ub) {
        matrixx_t A = matrixx_t::Zero(3*ncontacts_, nd_);

        pinocchio::forwardKinematics(model_.GetModel(), pin_data_, q, v, vectorx_t::Zero(nv_));

        for (int i = 0; i < in_contact.size(); i++) {
            if (in_contact[i]) {
                // Get the frame Jacobian
                matrix6_t J;
                model_.GetFrameJacobian(contact_frames_[i], q, J, pinocchio::LOCAL);
                A.block(3*i, 0, 3, nv_) = J.topRows<3>();

                // Get the Jdot term
                lb.segment<3>(3*i) = -pinocchio::getFrameAcceleration(model_.GetModel(), pin_data_,
                    model_.GetFrameIdx(contact_frames_[i]), pinocchio::LOCAL).linear().head<3>();
            }
        }
        // Equality constraint
        ub = lb;

        return A;
    }

    matrixx_t WbcController::ForceConstraint(const std::vector<bool> &in_contact, vectorx_t &lb, vectorx_t &ub) {
        matrixx_t A = matrixx_t::Zero(4*ncontacts_ + 3*(ncontact_frames_ - ncontacts_), nd_);

        int row_idx = 0;
        for (int i = 0; i < in_contact.size(); i++) {
            if (in_contact[i]) {
                // Linear friction cone
                A.block(row_idx, nv_ + ntau_ + 3*i, 4, 3) <<
                    -1, 0, mu_,
                    1, 0, mu_,
                    0, -1, mu_,
                    0, 1, mu_;

                lb.segment<4>(row_idx).setZero();
                ub.segment<4>(row_idx) = vector4_t::Constant(4, 1000);

                row_idx += 4;
            } else {
                A.block(row_idx, nv_ + ntau_ + 3*i, 3, 3).setIdentity();

                lb.segment<3>(row_idx).setZero();
                ub.segment<3>(row_idx).setZero();

                row_idx += 3;
            }
        }

        return A;
    }

    matrixx_t WbcController::DynamicsConstraint(const vectorx_t &q, const vectorx_t &v, vectorx_t& lb, vectorx_t& ub) {
        matrixx_t A(nv_, nd_);
        // Compute the terms via auto diff
        vectorx_t x_zero = vectorx_t::Zero(nd_);    // Linearize about 0, but it doesn't matter bc its already linear

        vectorx_t p(model_.GetConfigDim() + nv_);
        p << q , v;

        inverse_dynamics_->GetJacobian(x_zero, p, A);

        // Get the bounds
        vectorx_t y;
        inverse_dynamics_->GetFunctionValue(x_zero, p, y);

        lb = -y;
        ub = -y;

        return A;
    }

    void WbcController::DynamicsFunction(const ad::ad_vector_t &a_tau_F, const ad::ad_vector_t &q_v, ad::ad_vector_t &violation) {

        const ad::ad_vector_t& a = a_tau_F.head(nv_);
        const ad::ad_vector_t& tau = a_tau_F.segment(nv_, ntau_);
        const ad::ad_vector_t& F = a_tau_F.tail(nF_);

        const ad::ad_vector_t& q = q_v.head(model_.GetConfigDim());
        const ad::ad_vector_t& v = q_v.tail(nv_);

        // Make the force object
        std::vector<models::ExternalForce<ad::adcg_t>> f_ext;
        int idx = 0;
        for (const auto& frame : contact_frames_) {
            f_ext.emplace_back(frame, F.segment<CONTACT_3DOF>(idx));
            idx += CONTACT_3DOF;
        }

        // Call the inverse dynamics
        const ad::ad_vector_t tau_id = models::InverseDynamics(model_.GetADPinModel(), *model_.GetADPinData(),
            q, v, a, f_ext);

        violation = tau_id;
        violation.tail(ntau_) -= tau;
    }

    matrixx_t WbcController::TorqueBoxConstraint() const {
        matrixx_t A = matrixx_t::Zero(ntau_, nd_);
        A.block(0, nv_, ntau_, ntau_).setIdentity();

        return A;
    }


    std::pair<matrixx_t, vectorx_t> WbcController::StateTracking(const vectorx_t &q, const vectorx_t &v,
        const vectorx_t &q_des, const vectorx_t &v_des) {
        // Compute desired acceleration
        const vectorx_t q_diff = models::qDifference(q_des, q);
        const vectorx_t v_diff = v_des - v;

        const vectorx_t a_des = kp_.asDiagonal()*q_diff + kd_.asDiagonal()*v_diff;

        // Compute the mats
        matrixx_t P = matrixx_t::Zero(nv_, nv_);
        P.topLeftCorner<FLOATING_VEL, FLOATING_VEL>() = 2*base_weight_.asDiagonal();
        P.bottomRightCorner(nv_ - FLOATING_VEL, nv_ - FLOATING_VEL) = 2*joint_weight_.asDiagonal();

        vectorx_t g(nv_);
        g.head<FLOATING_VEL>().noalias() = base_weight_.asDiagonal()*-2*a_des.head<FLOATING_VEL>();
        g.tail(nv_ - FLOATING_VEL).noalias() = joint_weight_.asDiagonal()*-2*a_des.tail(nv_ - FLOATING_VEL);

        return {P, g};
    }

    std::pair<matrixx_t, vectorx_t> WbcController::TorqueTracking(const vectorx_t &tau_des) const {
        matrixx_t P = 2*tau_weight_.asDiagonal();
        vectorx_t g = tau_weight_.asDiagonal()*-2*tau_des;

        return {P, g};
    }

    std::pair<matrixx_t, vectorx_t> WbcController::ForceTracking(const vectorx_t &F_des) const {
        matrixx_t P = 2*force_weight_.asDiagonal();
        vectorx_t g = force_weight_.asDiagonal()*-2*F_des;

        return {P, g};
    }



}
