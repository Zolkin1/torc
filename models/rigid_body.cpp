//
// Created by zolkin on 5/23/24.
//

#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/contact-dynamics.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"

#include "rigid_body.h"

#include <utility>

namespace torc::models {
    RigidBody::RigidBody(std::string name, std::filesystem::path urdf)
        : PinocchioModel(std::move(name), std::move(urdf)) {
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
                                             const ContactState& contacts) const {
        assert(state.q.size() - 1 == state.v.size());

        matrixx_t J = ConstraintJacobian(contacts);
        vectorx_t gamma = ConstraintDrift(contacts);
        const vectorx_t& tau = InputsToFullTau(input);

        pinocchio::forwardDynamics(pin_model_, *pin_data_, state.q, state.v, tau,
                                          J, gamma);

        RobotStateDerivative xdot(state.v, pin_data_->ddq);

        return xdot;
    }

    void RigidBody::DynamicsDerivative(const RobotState& state, const vectorx_t& input,
                            matrixx_t& A, matrixx_t& B) const{
        assert(state.q.size() - 1 == state.v.size());

        const vectorx_t& tau = InputsToFullTau(input);

        // TODO: Do I need to account for external GRFs?
        pinocchio::computeABADerivatives(pin_model_, *pin_data_, state.q, state.v, tau);

        A << matrixx_t::Zero(pin_model_.nv, pin_model_.nq), matrixx_t::Identity(pin_model_.nv, pin_model_.nv),
                pin_data_->ddq_dq, pin_data_->ddq_dv;

        // Make into a full matrix
        pin_data_->Minv.triangularView<Eigen::StrictlyLower>() =
                pin_data_->Minv.transpose().triangularView<Eigen::StrictlyLower>();

        B << matrixx_t::Zero(pin_model_.nv, input.size()), pin_data_->Minv;
    }

    matrixx_t RigidBody::ConstraintJacobian(const ContactState& contacts) const {
        // TODO: Implement
        return matrixx_t::Zero(contacts.GetNumContacts()*3, pin_model_.nv);
    }

    vectorx_t RigidBody::ConstraintDrift(const ContactState& contacts) const {
        // TODO: Implement
        return vectorx_t ::Zero(contacts.GetNumContacts()*3);
    }
} // torc::models