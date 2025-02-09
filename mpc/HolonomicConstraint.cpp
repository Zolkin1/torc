//
// Created by zolkin on 1/19/25.
//

#include "include/HolonomicConstraint.h"

#include "pinocchio_interface.h"

namespace torc::mpc {

    HolonomicConstraint::HolonomicConstraint(int first_node, int last_node, const std::string &name,
        const models::FullOrderRigidBody &model, const std::vector<std::string> &contact_frames,
        const std::filesystem::path &deriv_lib_path, bool compile_derivs)
            : Constraint(first_node, last_node, name), model_(model) {

        for (const auto& frame : contact_frames) {
            constraint_functions_.emplace(frame,
                std::make_unique<ad::CppADInterface>(
                    std::bind(&HolonomicConstraint::HoloConstraint, this, frame, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                    name_ + "_" + frame + "_holonomic_constraint",
                    deriv_lib_path,
                    ad::DerivativeOrder::FirstOrder, 2*model_.GetVelDim(),  model_.GetConfigDim() + model_.GetVelDim(),
                    compile_derivs
                ));
        }
    }


    std::pair<matrixx_t, vectorx_t> HolonomicConstraint::GetLinearization(const vectorx_t& q_lin, const vectorx_t& v_lin,
            const std::string& frame) {
        vectorx_t x_zero = vectorx_t::Zero(constraint_functions_[frame]->GetDomainSize());
        vectorx_t p(constraint_functions_[frame]->GetParameterSize());

        p << q_lin, v_lin;

        matrixx_t jac;
        constraint_functions_[frame]->GetJacobian(x_zero, p, jac);

        vectorx_t y;
        constraint_functions_[frame]->GetFunctionValue(x_zero, p, y);

        return {jac, y};
    }

    int HolonomicConstraint::GetNumConstraints() const {
        return constraint_functions_.size()*constraint_functions_.begin()->second->GetRangeSize();
    }


    void HolonomicConstraint::HoloConstraint(const std::string& frame, const ad::ad_vector_t& dqk_dvk,
        const ad::ad_vector_t& qk_vk, ad::ad_vector_t& violation) {
        const ad::ad_vector_t& dq = dqk_dvk.head(model_.GetVelDim());
        const ad::ad_vector_t& dv = dqk_dvk.tail(model_.GetVelDim());

        const ad::ad_vector_t& qkbar = qk_vk.head(model_.GetConfigDim());
        const ad::ad_vector_t& vkbar = qk_vk.tail(model_.GetVelDim());

        const ad::ad_vector_t q = torc::models::ConvertdqToq(dq, qkbar);
        const ad::ad_vector_t v = vkbar + dv;

        // forward kinematics
        pinocchio::forwardKinematics(model_.GetADPinModel(), *model_.GetADPinData(), q, v);

        // Get the frame velocity
        const long frame_idx = model_.GetFrameIdx(frame);
        // TODO: Figure out which frame to use. I think either one should work
        const ad::ad_vector_t vel = pinocchio::getFrameVelocity(model_.GetADPinModel(), *model_.GetADPinData(), frame_idx, pinocchio::LOCAL).linear();

        // TODO: In the future we will want to rotate this into the ground frame so the constraint is always tangential to the terrain

        // Violation is the velocity as we want to drive it to 0
        violation = vel.head<2>();
    }

    vectorx_t HolonomicConstraint::GetViolation(const vectorx_t &q, const vectorx_t &v, const vectorx_t& dq,
        const vectorx_t& dv, const std::string &frame) {
        vectorx_t x(constraint_functions_[frame]->GetDomainSize());
        x << dq, dv;
        vectorx_t p(constraint_functions_[frame]->GetParameterSize());
        p << q, v;
        vectorx_t violation = vectorx_t::Zero(constraint_functions_[frame]->GetRangeSize());

        constraint_functions_[frame]->GetFunctionValue(x, p, violation);

        return violation;
    }

}
