//
// Created by zolkin on 1/30/25.
//

#include "pinocchio_interface.h"

#include "PolytopeConstraint.h"

namespace torc::mpc {
    PolytopeConstraint::PolytopeConstraint(int first_node, int last_node, const std::string &name,
        const std::vector<std::string> &contact_frames,
        const std::filesystem::path &deriv_lib_path, bool compile_derivs,
        const models::FullOrderRigidBody &model)
            : Constraint(first_node, last_node, name), model_(model), config_dim_(model.GetConfigDim()) {
        for (const auto& frame : contact_frames) {
            constraint_functions_.emplace(frame,
                std::make_unique<ad::CppADInterface>(
                    std::bind(&PolytopeConstraint::FootPolytopeConstraint, this, frame, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                    name_ + "_" + frame + "_holonomic_constraint",
                    deriv_lib_path,
                    ad::DerivativeOrder::FirstOrder, model_.GetVelDim(),  model_.GetConfigDim() + POLYTOPE_SIZE,
                    true //compile_derivs // TODO: Put back
                ));
        }
    }

    void PolytopeConstraint::GetLinearization(const vectorx_t &q_lin,
        const ContactInfo &contact_info, const std::string &frame,
        matrixx_t &lin_mat, vectorx_t &ub, vectorx_t &lb) {

        vectorx_t x_zero = vectorx_t::Zero(constraint_functions_[frame]->GetDomainSize());
        vectorx_t p(constraint_functions_[frame]->GetParameterSize());
        p << q_lin, contact_info.A_.row(0).transpose(), contact_info.A_.row(1).transpose();

        vectorx_t y;
        constraint_functions_[frame]->GetJacobian(x_zero, p, lin_mat);
        constraint_functions_[frame]->GetFunctionValue(x_zero, p, y);

        ub = contact_info.b_.head<2>() - y;
        lb = contact_info.b_.tail<2>() - y;
    }

    int PolytopeConstraint::GetNumConstraints() const {
        return constraint_functions_.size()*POLYTOPE_SIZE/2;
    }

    vectorx_t PolytopeConstraint::GetViolation(const vectorx_t &q_lin, const vectorx_t &dq,
        const ContactInfo &contact_info, const std::string &frame) {

        vectorx_t p(constraint_functions_[frame]->GetParameterSize());
        p << q_lin, contact_info.A_.row(0).transpose(), contact_info.A_.row(1).transpose();

        vectorx_t y;
        constraint_functions_[frame]->GetFunctionValue(dq, p, y);

        vectorx_t violation = vectorx_t::Zero(POLYTOPE_SIZE/2);

        if (y(0) > contact_info.b_(0)) {
            violation(0) = y(0) - contact_info.b_(0);
        } else if (y(0) < contact_info.b_(1)) {
            violation(0) = contact_info.b_(1) - y(0);
        }

        if (y(1) > contact_info.b_(2)) {
            violation(1) = y(1) - contact_info.b_(2);
        } else if (y(1) < contact_info.b_(3)) {
            violation(1) = contact_info.b_(3) - y(1);
        }

        return violation;
    }



    void PolytopeConstraint::FootPolytopeConstraint(const std::string &frame, const ad::ad_vector_t &dqk,
        const ad::ad_vector_t &qk_A, ad::ad_vector_t &poly_val) {

        const ad::ad_vector_t& qkbar = qk_A.head(config_dim_);
        ad::ad_matrix_t A = ad::ad_matrix_t::Zero(POLYTOPE_SIZE/2, 2);

        for (int i = 0; i < POLYTOPE_SIZE/2; i++) {
            A.row(i) = qk_A.segment<2>(config_dim_ + i*2).transpose();
        }

        const ad::ad_vector_t q = torc::models::ConvertdqToq(dqk, qkbar);

        // Forward kinematics
        pinocchio::forwardKinematics(model_.GetADPinModel(), *model_.GetADPinData(), q);
        const long frame_idx = model_.GetFrameIdx(frame);
        pinocchio::updateFramePlacement(model_.GetADPinModel(), *model_.GetADPinData(), frame_idx);

        // Get frame position in world frame (data oMf)
        ad::ad_vector_t frame_pos = model_.GetADPinData()->oMf.at(frame_idx).translation();

        poly_val.resize(POLYTOPE_SIZE/2);
        poly_val = A*frame_pos.head<2>();

        // violation << frame_pos.head<2>(), -frame_pos.head<2>();
        // violation = violation - b;

        // vector2_t x_temp;
        // x_temp << 1, 1;
        // violation = A*x_temp - b;

        // const ad::ad_vector_t& qkbar = qk_A_b.head(config_dim_);
        // ad::ad_matrix_t A = ad::ad_matrix_t::Zero(POLYTOPE_SIZE, 2);
        // // ad::ad_matrix_t A = ad::ad_matrix_t::Identity(POLYTOPE_SIZE, 2);
        //
        // for (int i = 0; i < POLYTOPE_SIZE/2; i++) {
        //     A.row(i) = qk_A_b.segment<2>(config_dim_ + i*2).transpose();
        // }
        //
        // A.bottomRows<POLYTOPE_SIZE/2>() = -A.topRows<POLYTOPE_SIZE/2>();
        //
        // ad::ad_vector_t b = qk_A_b.tail<POLYTOPE_SIZE>();        // ub then lb
        // b.tail<POLYTOPE_SIZE/2>() = -b.tail<POLYTOPE_SIZE/2>();
        //
        // const ad::ad_vector_t q = torc::models::ConvertdqToq(dqk, qkbar);
        //
        // // Forward kinematics
        // pinocchio::forwardKinematics(model_.GetADPinModel(), *model_.GetADPinData(), q);
        // const long frame_idx = model_.GetFrameIdx(frame);
        // pinocchio::updateFramePlacement(model_.GetADPinModel(), *model_.GetADPinData(), frame_idx);
        //
        // // Get frame position in world frame (data oMf)
        // ad::ad_vector_t frame_pos = model_.GetADPinData()->oMf.at(frame_idx).translation();
        //
        // violation.resize(POLYTOPE_SIZE);
        // violation = A*frame_pos.head<2>() - b;
        // // violation << frame_pos.head<2>(), -frame_pos.head<2>();
        // // violation = violation - b;
        //
        // // vector2_t x_temp;
        // // x_temp << 1, 1;
        // // violation = A*x_temp - b;
    }


}