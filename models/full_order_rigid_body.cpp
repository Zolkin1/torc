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
    FullOrderRigidBody::FullOrderRigidBody(const std::string& name, const std::filesystem::path& model_path, bool urdf_model)
        : PinocchioModel(name, model_path, HybridSystemImpulse, urdf_model) {

        // Assuming all joints (not "root_joint") are actuated
        std::vector<std::string> unactuated_joints;
        unactuated_joints.emplace_back("root_joint");

        CreateActuationMatrix(unactuated_joints);

        contact_data_ = std::make_unique<pinocchio::Data>(pin_model_);
    }

    FullOrderRigidBody::FullOrderRigidBody(const std::string& name, const std::filesystem::path& model_path,
                                           const std::vector<std::string>& underactuated_joints, bool urdf_model)
        : PinocchioModel(name, model_path, HybridSystemImpulse, urdf_model) {

        CreateActuationMatrix(underactuated_joints);

        contact_data_ = std::make_unique<pinocchio::Data>(pin_model_);
    }

    FullOrderRigidBody::FullOrderRigidBody(const torc::models::FullOrderRigidBody& other)
        : PinocchioModel(other.name_, other.model_path_, HybridSystemImpulse) {
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
        const vectorx_t tau = InputsToTau(input);
        pinocchio::aba(pin_model_, *pin_data_, q, v, tau);
        return BuildStateDerivative(v, pin_data_->ddq);
    }

    vectorx_t FullOrderRigidBody::GetDynamics(const vectorx_t& state,
                                              const vectorx_t& input,
                                              const RobotContactInfo& contact_info) const {
        vectorx_t q, v;
        ParseState(state, q, v);
        const vectorx_t tau = InputsToTau(input);

        // Create contact data
        // @Note that the RigidConstraint* classes are likely to change to just be generic constraints classes
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

    vectorx_t FullOrderRigidBody::InverseDynamics(const vectorx_t& q, const vectorx_t& v, const vectorx_t& a,
                                                  const std::vector<ExternalForce>& f_ext) {
//                                                  const pinocchio::container::aligned_vector<pinocchio::Force>& forces) {
        // Convert force to a pinocchio force
        pinocchio::container::aligned_vector<pinocchio::Force> forces = ConvertExternalForcesToPin(q, f_ext);

        pinocchio::rnea(this->pin_model_, *this->pin_data_, q, v, a, forces);

        vectorx_t tau = this->pin_data_->tau;
        return tau;
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

        const vectorx_t tau = InputsToTau(input);

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

        const vectorx_t tau = InputsToTau(input);

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

    void FullOrderRigidBody::InverseDynamicsDerivative(const torc::models::vectorx_t& state,
                                                       const torc::models::vectorx_t& a,
//                                                       const pinocchio::container::aligned_vector<pinocchio::Force>& forces,
                                                       const std::vector<ExternalForce>& f_ext,
                                                       torc::models::matrixx_t& dtau_dq,
                                                       torc::models::matrixx_t& dtau_dv,
                                                       torc::models::matrixx_t& dtau_da) {
        vectorx_t q, v;
        ParseState(state, q, v);
        assert(dtau_dq.rows() == GetVelDim());
        assert(dtau_dq.cols() == GetVelDim());
        assert(dtau_dv.rows() == GetVelDim());
        assert(dtau_dv.cols() == GetVelDim());
        assert(dtau_da.rows() == GetVelDim());
        assert(dtau_da.cols() == GetVelDim());
        assert(a.size() == v.size());

        pinocchio::container::aligned_vector<pinocchio::Force> forces = ConvertExternalForcesToPin(q, f_ext);

        pinocchio::computeRNEADerivatives(pin_model_, *pin_data_, q, v, a, forces, dtau_dq, dtau_dv, dtau_da);

        matrixx_t df_dq = ExternalForcesDerivativeWrtConfiguration(q, f_ext);

        dtau_dq = dtau_dq + df_dq;

        // I also need to account for how forces varies wrt q, i.e. I should add J(q)*dforces_dq to dtau_dq
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

    void FullOrderRigidBody::GetDynamicsTerms(const vectorx_t& state, matrixx_t& M, matrixx_t& C, vectorx_t& g) {
        vectorx_t q, v;
        ParseState(state, q, v);

        this->SecondOrderFK(q, v);

        pinocchio::crba(pin_model_, *pin_data_, q);

        // Get M and make it symmetric
        M = pin_data_->M;
        M.triangularView<Eigen::StrictlyLower>() =
                M.transpose().triangularView<Eigen::StrictlyLower>();

        // Get C
        pinocchio::computeCoriolisMatrix(pin_model_, *pin_data_, q, v);
        C = pin_data_->C;

        // Get g
        pinocchio::computeGeneralizedGravity(pin_model_, *pin_data_, q);
        g = pin_data_->g;
    }

    pinocchio::Motion FullOrderRigidBody::GetFrameAcceleration(const std::string& frame) {
        return pinocchio::getFrameClassicalAcceleration(pin_model_, *pin_data_, this->GetFrameIdx(frame), pinocchio::LOCAL_WORLD_ALIGNED);
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

    pinocchio::container::aligned_vector<pinocchio::Force> FullOrderRigidBody::ConvertExternalForcesToPin(const vectorx_t& q,
            const std::vector<ExternalForce>& f_ext) const {
        pinocchio::forwardKinematics(pin_model_, *pin_data_, q);
        pinocchio::framesForwardKinematics(pin_model_, *pin_data_, q);

        // Convert force to a pinocchio force
        pinocchio::container::aligned_vector<pinocchio::Force> forces(this->GetNumJoints(), pinocchio::Force::Zero());
        for (const auto& f : f_ext) {
            // *** Note *** for now I only support 3DOF contacts. To support 6DOF contacts just need to add the additional torques in from the contact (similar to how the linear forces are translated)
            // Get the frame where the contact is
            const int frame_idx = this->GetFrameIdx(f.frame_name);
            // Get the parent frame
            const int jnt_idx = this->pin_model_.frames.at(frame_idx).parentJoint;

            // Get the translation from the joint frame to the contact frame
            const vector3_t translationContactToJoint = pin_model_.frames.at(frame_idx).placement.translation();

            // Get the rotation from the world frame to the joint frame
            const Eigen::Matrix3d rotationWorldToJoint = pin_data_->oMi[jnt_idx].rotation().transpose();

            // Get the contact forces in the joint frame
            const vector3_t contact_force = rotationWorldToJoint * f.force_linear;
            forces.at(jnt_idx).linear() = contact_force;

            // Calculate the angular (torque) forces
            forces.at(jnt_idx).angular() = translationContactToJoint.cross(contact_force);
        }
        return forces;
    }

    matrixx_t FullOrderRigidBody::ExternalForcesDerivativeWrtConfiguration(const vectorx_t& q, const std::vector<ExternalForce>& f_ext) {
        // sum of J(q) * df_dq
        // J(q) are the frame jacobians for the joints
        // df_dq is the derivative of ConvertExternalForcesToPin
        pinocchio::forwardKinematics(pin_model_, *pin_data_, q);
        pinocchio::framesForwardKinematics(pin_model_, *pin_data_, q);

//        matrixx_t df_dq = matrixx_t::Zero(this->GetVelDim(), this->GetVelDim());
        matrix6x_t df_dq = matrixx_t::Zero(6, this->GetVelDim());
        for (const auto& f : f_ext) {
            // Get the contact frame
            const int frame_idx = this->GetFrameIdx(f.frame_name);
            // Get the parent frame
            const int jnt_idx = this->pin_model_.frames.at(frame_idx).parentJoint;
            const vector3_t translationContactToJoint = pin_model_.frames.at(frame_idx).placement.translation();

            // Get the contact Jacobian
//            matrix6x_t J = matrix6x_t::Zero(6, this->GetVelDim());
//            pinocchio::computeFrameJacobian(pin_model_, *pin_data_, q, frame_idx, pinocchio::LOCAL_WORLD_ALIGNED, J);

            // Get the derivative of joint frame wrt q (i.e. a frame jacobian)
            std::cout << "true joint name: " << this->pin_model_.names.at(jnt_idx) << std::endl;
            int joint_frame_idx = this->GetFrameIdx(this->pin_model_.names.at(jnt_idx));
            std::cout << "joint frame: " << this->pin_model_.frames.at(joint_frame_idx).name << std::endl;
            std::cout << "torso frame: " << this->pin_model_.frames.at(GetFrameIdx("base")).name << std::endl;

            matrix6x_t joint_frame_jacobian = matrix6x_t::Zero(6, this->GetVelDim());
            pinocchio::computeFrameJacobian(pin_model_, *pin_data_, q, joint_frame_idx, pinocchio::LOCAL_WORLD_ALIGNED, joint_frame_jacobian);

            matrix6x_t joint_frame_jacobian_world = matrix6x_t::Zero(6, this->GetVelDim());
            pinocchio::computeFrameJacobian(pin_model_, *pin_data_, q, joint_frame_idx, pinocchio::WORLD, joint_frame_jacobian_world);

            matrix6x_t torso_frame_jacobian_world = matrix6x_t::Zero(6, this->GetVelDim());
            pinocchio::computeFrameJacobian(pin_model_, *pin_data_, q, GetFrameIdx("base"), pinocchio::LOCAL_WORLD_ALIGNED, torso_frame_jacobian_world);

            std::cout << "Joint frame jacobian: \n" << joint_frame_jacobian << std::endl;
            std::cout << "Joint frame jacobian wrt world: \n" << joint_frame_jacobian_world << std::endl;
            std::cout << "Torso frame jacobian wrt world: \n" << torso_frame_jacobian_world << std::endl;

            // Get the rotation from the world frame to the joint frame
            const Eigen::Matrix3d rotationWorldToJoint = pin_data_->oMi[jnt_idx].rotation().transpose();

            matrix6x_t dFdq = matrix6x_t::Zero(6, this->GetVelDim());
            dFdq.topRows<3>() = joint_frame_jacobian_world.bottomRows<3>(); //joint_frame_jacobian_world.bottomRows<3>();
            for (int i = 0; i < this->GetVelDim(); i++) {
                dFdq.block<3,1>(3, i) = translationContactToJoint.cross(joint_frame_jacobian_world.block<3,1>(3, i));
            }

            std::cout << "dFdq: \n" << dFdq << std::endl;

            // TODO: I have the correct sparsity pattern and same general patterns, but the numbers are off
//            df_dq = df_dq - joint_frame_jacobian.transpose()*dFdq;
            df_dq = dFdq; // For now we just want to compare it against the function
        }

        return df_dq;
    }

    vectorx_t FullOrderRigidBody::GetUpperConfigLimits() const {
        return pin_model_.upperPositionLimit;
    }

    vectorx_t FullOrderRigidBody::GetLowerConfigLimits() const {
        return pin_model_.lowerPositionLimit;
    }

    vectorx_t FullOrderRigidBody::GetVelocityJointLimits() const {
        return pin_model_.velocityLimit;
    }

    vectorx_t FullOrderRigidBody::GetTorqueJointLimits() const {
        return pin_model_.effortLimit.tail(GetNumInputs());
    }




//    vectorx_t FullOrderRigidBody::TauToInputs(const vectorx_t& tau) const {
//        assert(tau == this->GetNumJoints());
//        return
//    }
} // torc::models
