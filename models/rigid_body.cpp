//
// Created by zolkin on 5/23/24.
//

#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/constrained-dynamics.hpp"
#include "pinocchio/algorithm/constrained-dynamics-derivatives.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"

#include "rigid_body.h"

#include <utility>

namespace torc::models {
    RigidBody::RigidBody(std::string name, std::filesystem::path urdf, const RobotContactInfo& contact_info)
        : PinocchioModel(std::move(name), std::move(urdf), contact_info) {
        // TODO: Set size of J_ and gamma_
    }

    RobotStateDerivative RigidBody::GetDynamics(const RobotState& state, const vectorx_t& input) const {
        assert(state.q.size() - 1 == state.v.size());

        const vectorx_t& tau = InputsToFullTau(input);

        pinocchio::aba(pin_model_, *pin_data_, state.q, state.v, tau);

        RobotStateDerivative xdot(state.v, pin_data_->ddq);

        return xdot;
    }

    RobotStateDerivative RigidBody::GetDynamics(const RobotState& state, const vectorx_t& input,
                                             const RobotContactInfo& contacts) const {
        assert(state.q.size() - 1 == state.v.size());

        matrixx_t J = ConstraintJacobian(contacts);
        vectorx_t gamma = ConstraintDrift(contacts);
        const vectorx_t& tau = InputsToFullTau(input);

        // Convert the contacts into pinocchio contacts

        // Call initConstraintDynamics

        // Call constraintDynamics

//        pinocchio::forwardDynamics(pin_model_, *pin_data_, state.q, state.v, tau,
//                                          J, gamma);

        RobotStateDerivative xdot(state.v, pin_data_->ddq);

        return xdot;
    }

    void RigidBody::DynamicsDerivative(const RobotState& state, const vectorx_t& input,
                            matrixx_t& A, matrixx_t& B) const{
        assert(state.q.size() - 1 == state.v.size());
        assert(A.rows() == GetDerivativeDim());
        assert(A.cols() == GetDerivativeDim());
        assert(B.rows() == GetDerivativeDim());
        assert(B.cols() == act_mat_.cols());

        const vectorx_t& tau = InputsToFullTau(input);

        pinocchio::computeABADerivatives(pin_model_, *pin_data_, state.q, state.v, tau);

        A << matrixx_t::Zero(pin_model_.nv, pin_model_.nv), matrixx_t::Identity(pin_model_.nv, pin_model_.nv),
                pin_data_->ddq_dq, pin_data_->ddq_dv;

        // Make into a full matrix
        pin_data_->Minv.triangularView<Eigen::StrictlyLower>() =
                pin_data_->Minv.transpose().triangularView<Eigen::StrictlyLower>();

        B << matrixx_t::Zero(pin_model_.nv, input.size()), pin_data_->Minv * act_mat_;
    }

    matrixx_t RigidBody::ConstraintJacobian(const RobotContactInfo& contacts) const {
        // TODO: Implement
        return matrixx_t::Zero(contacts.GetNumContacts()*3, pin_model_.nv);
    }

    vectorx_t RigidBody::ConstraintDrift(const RobotContactInfo& contacts) const {
        // TODO: Implement
        return vectorx_t ::Zero(contacts.GetNumContacts()*3);
    }
} // torc::models