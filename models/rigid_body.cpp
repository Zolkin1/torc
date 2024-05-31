//
// Created by zolkin on 5/23/24.
//

#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/constrained-dynamics.hpp"
#include "pinocchio/algorithm/constrained-dynamics-derivatives.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/impulse-dynamics.hpp"
#include "pinocchio/algorithm/impulse-dynamics-derivatives.hpp"
#include "pinocchio/algorithm/proximal.hpp"

#include "rigid_body.h"

#include <utility>

namespace torc::models {
    RigidBody::RigidBody(std::string name, std::filesystem::path urdf)
        : PinocchioModel(std::move(name), std::move(urdf)) {
    }

    RobotStateDerivative RigidBody::GetDynamics(const RobotState& state, const vectorx_t& input) const {
        assert(state.q.size() - 1 == state.v.size());

        const vectorx_t& tau = InputsToFullTau(input);

        pinocchio::aba(pin_model_, *pin_data_, state.q, state.v, tau);

        RobotStateDerivative xdot(state.v, pin_data_->ddq);

        return xdot;
    }

    RobotStateDerivative RigidBody::GetDynamics(const RobotState& state, const vectorx_t& input,
                                             const RobotContactInfo& contact_info) const {
        assert(state.q.size() - 1 == state.v.size());

        const vectorx_t& tau = InputsToFullTau(input);

        // Create contact data
        // TODO: Note that the RigidConstraint* classes are likely to change to just be generic constraint classes
        //  when the pinocchio 3 api is more stabilized. For now this is what we have.
        std::vector<pinocchio::RigidConstraintModel> contact_model;
        std::vector<pinocchio::RigidConstraintData> contact_data;

        MakePinocchioContacts(contact_info, contact_model, contact_data);

        // Call initConstraintDynamics
        pinocchio::initConstraintDynamics(pin_model_, *pin_data_, contact_model);

        // Call constraintDynamics
        pinocchio::constraintDynamics(pin_model_, *pin_data_, state.q, state.v, tau, contact_model, contact_data);

        RobotStateDerivative xdot(state.v, pin_data_->ddq);

        return xdot;
    }

    RobotState RigidBody::GetImpulseDynamics(const RobotState& state, const vectorx_t& input,
                                                  const RobotContactInfo& contact_info) const {
        // Create contact data
        std::vector<pinocchio::RigidConstraintModel> contact_model;
        std::vector<pinocchio::RigidConstraintData> contact_data;

        MakePinocchioContacts(contact_info, contact_model, contact_data);

        // initConstraintDynamics
        pinocchio::initConstraintDynamics(pin_model_, *pin_data_, contact_model);

        // impulseDynamics
        const pinocchio::ProximalSettings settings;     // Proximal solver settings. Set to default
        const double eps = 0;                           // Coefficient of restitution
        pinocchio::impulseDynamics(pin_model_, *pin_data_, state.q, state.v, contact_model,
                                   contact_data, eps, settings);

        const RobotState x(state.q, pin_data_->dq_after);
        return x;

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

    void RigidBody::ImpulseDerivative(const torc::models::RobotContactInfo& contact_info,
                                      matrixx_t& A, matrixx_t& B) const {
        assert(A.rows() == GetDerivativeDim());
        assert(A.cols() == GetDerivativeDim());
        assert(B.rows() == GetDerivativeDim());
        assert(B.cols() == act_mat_.cols());

        // Create contact data
        std::vector<pinocchio::RigidConstraintModel> contact_model;
        std::vector<pinocchio::RigidConstraintData> contact_data;

        MakePinocchioContacts(contact_info, contact_model, contact_data);

        // initConstraintDynamics
        pinocchio::initConstraintDynamics(pin_model_, *pin_data_, contact_model);

        // impulseDynamicsDerivatives
        const pinocchio::ProximalSettings settings;     // Proximal solver settings. Set to default
        const double eps = 0;                           // Coefficient of restitution
        pinocchio::computeImpulseDynamicsDerivatives(pin_model_, *pin_data_, contact_model,
                                   contact_data, eps, settings);

        A << matrixx_t::Zero(pin_model_.nv, pin_model_.nv), matrixx_t::Identity(pin_model_.nv, pin_model_.nv),
                pin_data_->ddq_dq, pin_data_->ddq_dv;

        B.setZero();
    }
} // torc::models