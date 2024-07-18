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

#include "full_order_rigid_body.h"

namespace torc::models {
    FullOrderRigidBody::FullOrderRigidBody(const std::string& name, const std::filesystem::path& urdf)
        : PinocchioModel(name, urdf, HybridSystemImpulse) {

        // Assuming all joints (not "root_joint") are actuated
        std::vector<std::string> unactuated_joints;
        unactuated_joints.emplace_back("root_joint");

        CreateActuationMatrix(unactuated_joints);

        contact_data_ = std::make_unique<pinocchio::Data>(pin_model_);
    }

    FullOrderRigidBody::FullOrderRigidBody(const std::string& name, const std::filesystem::path& urdf,
                                           const std::vector<std::string>& underactuated_joints)
        : PinocchioModel(name, urdf, HybridSystemImpulse) {

        CreateActuationMatrix(underactuated_joints);

        contact_data_ = std::make_unique<pinocchio::Data>(pin_model_);
    }

    FullOrderRigidBody::FullOrderRigidBody(const torc::models::FullOrderRigidBody& other)
        : PinocchioModel(other.name_, other.urdf_, HybridSystemImpulse) {
        n_input_ = other.n_input_;

        act_mat_ = other.act_mat_;
        contact_data_ = std::make_unique<pinocchio::Data>(*other.contact_data_);
    }

    int FullOrderRigidBody::GetStateDim() const {
        return this->GetConfigDim() + this->GetVelDim();
    }

    int FullOrderRigidBody::GetDerivativeDim() const {
        return 2*this->GetVelDim();
    }

    vectorx_t FullOrderRigidBody::GetRandomState() const {
        vectorx_t x(GetStateDim());
        x << GetRandomConfig(), GetRandomVel();
        return x;
    }

    quat_t FullOrderRigidBody::GetBaseOrientation(const vectorx_t& q) const {
        return static_cast<quat_t>(q.segment<4>(3));
    }

    vectorx_t FullOrderRigidBody::GetDynamics(const vectorx_t& state, const vectorx_t& input) {
        vectorx_t q, v;
        ParseState(state, q, v);
        const vectorx_t& tau = InputsToTau(input);
        pinocchio::aba(pin_model_, *pin_data_, q, v, tau);
        return BuildStateDerivative(v, pin_data_->ddq);
    }

    vectorx_t FullOrderRigidBody::GetDynamics(const vectorx_t& state,
                                              const vectorx_t& input,
                                              const RobotContactInfo& contact_info) const {
        vectorx_t q, v;
        ParseState(state, q, v);
        const vectorx_t& tau = InputsToTau(input);

        // Create contact data
        // @Note that the RigidConstraint* classes are likely to change to just be generic constraint classes
        //  when the pinocchio 3 api is more stabilized. For now this is what we have.
        std::vector<pinocchio::RigidConstraintModel> contact_model;
        std::vector<pinocchio::RigidConstraintData> contact_data;

        MakePinocchioContacts(contact_info, contact_model, contact_data);

        // Call initConstraintDynamics
        pinocchio::initConstraintDynamics(pin_model_, *contact_data_, contact_model);

        pinocchio::constraintDynamics(pin_model_, *contact_data_, q, v, tau,
                                      contact_model, contact_data);
        return BuildStateDerivative(v, contact_data_->ddq);
    }

    vectorx_t FullOrderRigidBody::GetImpulseDynamics(const vectorx_t& state,
                                                     const vectorx_t& input,
                                                     const RobotContactInfo& contact_info) {
        vectorx_t q, v;
        ParseState(state, q, v);
        // Create contact data
        std::vector<pinocchio::RigidConstraintModel> contact_model;
        std::vector<pinocchio::RigidConstraintData> contact_data;

        MakePinocchioContacts(contact_info, contact_model, contact_data);

        // initConstraintDynamics
        pinocchio::initConstraintDynamics(pin_model_, *contact_data_, contact_model);

        // impulseDynamics
        const pinocchio::ProximalSettings settings;         // Proximal solver settings. Set to default
        constexpr double eps = 0;                           // Coefficient of restitution
        pinocchio::impulseDynamics(pin_model_, *contact_data_, q, v, contact_model,
                                   contact_data, eps, settings);

        return BuildState(q, contact_data_->dq_after);
    }

    void FullOrderRigidBody::DynamicsDerivative(const vectorx_t& state, const vectorx_t& input,
                                                matrixx_t& A, matrixx_t& B) {
        vectorx_t q, v;
        ParseState(state, q, v);
        assert(A.rows() == GetDerivativeDim());
        assert(A.cols() == GetDerivativeDim());
        assert(B.rows() == GetDerivativeDim());
        assert(B.cols() == act_mat_.cols());

        const vectorx_t& tau = InputsToTau(input);

        pinocchio::computeABADerivatives(pin_model_, *pin_data_, q, v, tau);

        A << matrixx_t::Zero(pin_model_.nv, pin_model_.nv),
                matrixx_t::Identity(pin_model_.nv, pin_model_.nv),
                pin_data_->ddq_dq, pin_data_->ddq_dv;

        // Make into a full matrix
        pin_data_->Minv.triangularView<Eigen::StrictlyLower>() =
                pin_data_->Minv.transpose().triangularView<Eigen::StrictlyLower>();

        B << matrixx_t::Zero(pin_model_.nv, input.size()), pin_data_->Minv * act_mat_;
    }

    void FullOrderRigidBody::DynamicsDerivative(const vectorx_t& state, const vectorx_t& input,
                                                const RobotContactInfo& contacts,
                                                matrixx_t& A, matrixx_t& B) {
        vectorx_t q, v;
        ParseState(state, q, v);
        assert(A.rows() == GetDerivativeDim());
        assert(A.cols() == GetDerivativeDim());
        assert(B.rows() == GetDerivativeDim());
        assert(B.cols() == act_mat_.cols());

        const vectorx_t& tau = InputsToTau(input);

        // Create contact data
        std::vector<pinocchio::RigidConstraintModel> contact_model;
        std::vector<pinocchio::RigidConstraintData> contact_data;

        MakePinocchioContacts(contacts, contact_model, contact_data);

        // Call initConstraintDynamics
        pinocchio::initConstraintDynamics(pin_model_, *contact_data_, contact_model);

        pinocchio::constraintDynamics(pin_model_, *contact_data_, q, v, tau,
                                      contact_model, contact_data);

        pinocchio::ProximalSettings settings;     // Proximal solver settings. Set to default
        pinocchio::computeConstraintDynamicsDerivatives(pin_model_, *contact_data_,
                                                        contact_model, contact_data,
                                                        settings);

        A << matrixx_t::Zero(pin_model_.nv, pin_model_.nv),
                matrixx_t::Identity(pin_model_.nv, pin_model_.nv),
                contact_data_->ddq_dq, contact_data_->ddq_dv;

        // Make into a full matrix
        contact_data_->Minv.triangularView<Eigen::StrictlyLower>() =
                contact_data_->Minv.transpose().triangularView<Eigen::StrictlyLower>();

        B << matrixx_t::Zero(pin_model_.nv, input.size()), contact_data_->ddq_dtau * act_mat_; //contact_data_->Minv * act_mat_;
    }

    void FullOrderRigidBody::ImpulseDerivative(const vectorx_t& state, const vectorx_t& input,
                                               const RobotContactInfo& contact_info,
                                               matrixx_t& A, matrixx_t& B) {
        vectorx_t q, v;
        ParseState(state, q, v);
        assert(A.rows() == GetDerivativeDim());
        assert(A.cols() == GetDerivativeDim());
        assert(B.rows() == GetDerivativeDim());
        assert(B.cols() == act_mat_.cols());

        GetImpulseDynamics(state, input, contact_info);

        // Create contact data
        std::vector<pinocchio::RigidConstraintModel> contact_model;
        std::vector<pinocchio::RigidConstraintData> contact_data;

        MakePinocchioContacts(contact_info, contact_model, contact_data);

        // initConstraintDynamics
        pinocchio::initConstraintDynamics(pin_model_, *contact_data_, contact_model);

        const pinocchio::ProximalSettings settings;     // Proximal solver settings. Set to default
        constexpr double eps = 0;                           // Coefficient of restitution
        pinocchio::impulseDynamics(pin_model_, *contact_data_, q, v,
                                   contact_model, contact_data, eps, settings);

        // impulseDynamicsDerivatives
        pinocchio::computeImpulseDynamicsDerivatives(pin_model_, *contact_data_, contact_model,
                                   contact_data, eps, settings);

        A << matrixx_t::Identity(pin_model_.nv, pin_model_.nv),
                matrixx_t::Zero(pin_model_.nv, pin_model_.nv),
                contact_data_->ddq_dq, contact_data_->ddq_dv + matrixx_t::Identity(pin_model_.nv, pin_model_.nv);

        B.setZero();
    }

    void FullOrderRigidBody::CreateActuationMatrix(
        const std::vector<std::string> &underactuated_joints) {
      assert(pin_model_.idx_vs.at(1) == 0);
      assert(pin_model_.nvs.at(1) == FLOATING_VEL);

      int num_actuators = pin_model_.nv;
      const int num_joints = pin_model_.njoints;

      std::vector<int> unact_joint_idx;

      unact_joint_idx.push_back(0); // Universe joint is never actuated

      // Get the number of actuators
      for (const std::string& joint_name : underactuated_joints) {
        for (int i = 0; i < num_joints; i++) {
          if (joint_name == pin_model_.names.at(i)) {
            num_actuators -= pin_model_.joints.at(i).nv();
            unact_joint_idx.push_back(i);
            break;
          }
        }
      }

      act_mat_ = matrixx_t::Zero(pin_model_.nv, num_actuators);
      int act_idx = 0;

      for (int joint_idx = 0; joint_idx < num_joints; joint_idx++) {
        bool act = true;
        for (int idx : unact_joint_idx) {
          if (joint_idx == idx) {
            act = false;
            break;
          }
        }

        if (act) {
          const int nv = pin_model_.joints.at(joint_idx).nv();
          act_mat_.block(pin_model_.joints.at(joint_idx).idx_v(), act_idx, nv,
                         nv) = matrixx_t::Identity(nv, nv);
          act_idx += nv;
        }
      }

      n_input_ = act_mat_.cols();
    }

    void FullOrderRigidBody::ParseState(const vectorx_t &state,
                                        vectorx_t &q, vectorx_t &v) const {
        q = state.topRows(pin_model_.nq);
        v = state.bottomRows(pin_model_.nv);
    }

    void FullOrderRigidBody::ParseStateDerivative(const vectorx_t &dstate,
                                                  vectorx_t &v,
                                                  vectorx_t &a) const {
        v = dstate.topRows(pin_model_.nv);
        a = dstate.bottomRows(pin_model_.nv);
    }

    vectorx_t FullOrderRigidBody::BuildState(const vectorx_t &q, const vectorx_t &v) {
      vectorx_t x(q.size() + v.size());
      x << q, v;
      return x;
    }

    vectorx_t FullOrderRigidBody::BuildStateDerivative(const vectorx_t &v,
                                                       const vectorx_t &a) {
        vectorx_t x(v.size() + a.size());
        x << v, a;
        return x;
    }

    void FullOrderRigidBody::ParseInput(const vectorx_t &input, vectorx_t &tau) const {
      tau = act_mat_ * input;
    }

    vectorx_t FullOrderRigidBody::InputsToTau(const vectorx_t& input) const {
        assert(input.size() == act_mat_.cols());
        return act_mat_*input;
    }
} // torc::models