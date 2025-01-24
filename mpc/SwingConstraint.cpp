//
// Created by zolkin on 1/18/25.
//

#include "pinocchio/algorithm/frames.hpp"

#include "SwingConstraint.h"
#include "pinocchio_interface.h"

namespace torc::mpc {
    SwingConstraint::SwingConstraint(int first_node, int last_node, const std::string& name,
        const models::FullOrderRigidBody& model, const std::vector<std::string>& contact_frames,
        const std::filesystem::path& deriv_lib_path, bool compile_derivs)
        : Constraint(first_node, last_node, name), model_(model), contact_frames_(contact_frames) {

        for (const auto& frame : contact_frames) {
            swing_function_.emplace(frame,
               std::make_unique<ad::CppADInterface>(
                   std::bind(&SwingConstraint::SwingHeightConstraint, this, frame, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                  name_ + "_" + frame + "_swing_height_constraint",
                  deriv_lib_path,
                  ad::DerivativeOrder::FirstOrder, model_.GetVelDim(),  model_.GetConfigDim() + 1,
                  compile_derivs
               ));
        }
    }

    std::pair<matrixx_t, vectorx_t> SwingConstraint::GetLinearization(const vectorx_t &q_lin, double des_height,
        const std::string &frame) {
        vectorx_t x_zero = vectorx_t::Zero(swing_function_[frame]->GetDomainSize());
        vectorx_t p(swing_function_[frame]->GetParameterSize());
        p << q_lin, des_height;

        matrixx_t jac;
        swing_function_[frame]->GetJacobian(x_zero, p, jac);

        vectorx_t y;
        swing_function_[frame]->GetFunctionValue(x_zero, p, y);

        return {jac, y};
    }

    int SwingConstraint::GetNumConstraints() const {
        return swing_function_.size()*swing_function_.begin()->second->GetRangeSize();
    }

    void SwingConstraint::SwingHeightConstraint(const std::string& frame, const ad::ad_vector_t& dqk,
        const ad::ad_vector_t& qk_desheight, ad::ad_vector_t& violation) {
        const ad::ad_vector_t& qkbar = qk_desheight.head(model_.GetConfigDim());
        const ad::adcg_t& des_height = qk_desheight(model_.GetConfigDim());

        const ad::ad_vector_t q = torc::models::ConvertdqToq(dqk, qkbar);

        // Forward kinematics
        pinocchio::forwardKinematics(model_.GetADPinModel(), *model_.GetADPinData(), q);
        const long frame_idx = model_.GetFrameIdx(frame);
        pinocchio::updateFramePlacement(model_.GetADPinModel(), *model_.GetADPinData(), frame_idx);

        // Get frame position in world frame (data oMf)
        ad::ad_vector_t frame_pos = model_.GetADPinData()->oMf.at(frame_idx).translation();

        violation.resize(1);
        violation(0) = frame_pos(2) - des_height;
    }

    vectorx_t SwingConstraint::GetViolation(const vectorx_t &q, const vectorx_t& dq, double des_height, const std::string& frame) {
        vectorx_t x(swing_function_[frame]->GetDomainSize());
        x << dq;

        vectorx_t p(swing_function_[frame]->GetParameterSize());
        p << q, des_height;

        vectorx_t violation;
        swing_function_[frame]->GetFunctionValue(x, p, violation);
        return violation;
    }

}
