//
// Created by zolkin on 5/23/24.
//

#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/contact-dynamics.hpp"

#include "rigid_body.h"

#include <utility>

namespace torc::models {
    RigidBody::RigidBody(std::string name, std::filesystem::path urdf)
        : PinocchioModel(std::move(name), std::move(urdf)) {

    }

    vectorx_t RigidBody::GetDynamics(const RobotState& state, const vectorx_t& input) const {
        // TODO: Should I be using threaded version?
        return pinocchio::aba(pin_model_, *pin_data_, state.GetRefq(), state.GetRefv(), input);
    }

    vectorx_t RigidBody::GetDynamicsContacts(const RobotState& state, const vectorx_t& input,
                                             const ContactState& contacts) const {

        const matrixx_t J = ConstraintJacobian(contacts);
        const vectorx_t gamma = ConstraintDrift(contacts);

        pinocchio::forwardDynamics(pin_model_, *pin_data_, state.GetRefq(), state.GetRefv(), input,
                                          J, gamma);

        return pin_data_->ddq;
    }

    matrixx_t RigidBody::Dfdx(const RobotState& state, const vectorx_t& input) const {

    }

    matrixx_t RigidBody::Dfdu(const RobotState& state, const vectorx_t& input) const {

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