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
#include "pinocchio/algorithm/joint-configuration.hpp"

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

    FullOrderRigidBody::FullOrderRigidBody(const std::string& name, const std::filesystem::path& model_path,
            const std::vector<std::string>& joint_skip_names, const std::vector<double>& joint_skip_values)
        : PinocchioModel(name, model_path, HybridSystemImpulse, joint_skip_names, joint_skip_values) {
        std::vector<std::string> unactuated_joints;
        unactuated_joints.emplace_back("root_joint");

        CreateActuationMatrix(unactuated_joints);

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

    vectorx_t FullOrderRigidBody::IntegrateVelocity(const torc::models::vectorx_t& q0,
                                                    const torc::models::vectorx_t& v) const {
        return pinocchio::integrate(pin_model_, q0, v);
    }

    vectorx_t FullOrderRigidBody::GetDynamics(const vectorx_t& state, const vectorx_t& input) {
        vectorx_t q, v;
        ParseState(state, q, v);
        const vectorx_t tau = InputsToTau(input);
        pinocchio::aba(pin_model_, *pin_data_, q, v, tau);
        return BuildStateDerivative(v, pin_data_->ddq);
    }

    vectorx_t FullOrderRigidBody::GetDynamics(const vectorx_t& q, const vectorx_t& v, const vectorx_t& input,
                                              const std::vector<ExternalForce<double>>& f_ext) {
        const vectorx_t tau = InputsToTau(input);
        pinocchio::container::aligned_vector<pinocchio::Force> forces = ConvertExternalForcesToPin(q, f_ext);
        pinocchio::aba(pin_model_, *pin_data_, q, v, tau, forces);
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
                                                  const std::vector<ExternalForce<double>>& f_ext) {
//                                                  const pinocchio::container::aligned_vector<pinocchio::Force>& forces) {
        // Convert force to a pinocchio force
        pinocchio::container::aligned_vector<pinocchio::Force> forces = ConvertExternalForcesToPin(q, f_ext);

        return pinocchio::rnea(this->pin_model_, *this->pin_data_, q, v, a, forces);
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

    // TODO: when I have two contacts on the same link I seem to get a mismatch with the finite diff
    void FullOrderRigidBody::InverseDynamicsDerivative(const vectorx_t& q,
                                                       const vectorx_t& v,
                                                       const vectorx_t& a,
//                                                       const pinocchio::container::aligned_vector<pinocchio::Force>& forces,
                                                       const std::vector<ExternalForce<double>>& f_ext,
                                                       matrixx_t& dtau_dq,
                                                       matrixx_t& dtau_dv,
                                                       matrixx_t& dtau_da,
                                                       matrixx_t& dtau_df) {
        assert(dtau_dq.rows() == GetVelDim());
        assert(dtau_dq.cols() == GetVelDim());
        assert(dtau_dv.rows() == GetVelDim());
        assert(dtau_dv.cols() == GetVelDim());
        assert(dtau_da.rows() == GetVelDim());
        assert(dtau_da.cols() == GetVelDim());
        assert(dtau_df.cols() == f_ext.size()*3);
        assert(dtau_df.rows() == GetVelDim());
        assert(a.size() == v.size());

        pinocchio::container::aligned_vector<pinocchio::Force> forces = ConvertExternalForcesToPin(q, f_ext);

        pinocchio::computeRNEADerivatives(pin_model_, *pin_data_, q, v, a, dtau_dq, dtau_dv, dtau_da); // forces

        // dtau_df is the partial of tau wrt the linear external forces at the contact points in the world frame
        // so dtau_df is vel_dim X 3*contact_points
        // Each force enters through a frame jacobian
        matrix6x_t jacobian = matrix6x_t::Zero(6, GetVelDim());
        int df_idx = 0;
        // No need to call computeJointJacobians first because rneaDerivatives internally updates all of this
        for (const auto& f : f_ext) {
            pinocchio::getFrameJacobian(pin_model_, *pin_data_, GetFrameIdx(f.frame_name), pinocchio::LOCAL_WORLD_ALIGNED, jacobian);
            dtau_df.middleCols<3>(df_idx) = -jacobian.topRows<3>().transpose();
            jacobian.setZero();
            df_idx += 3;
        }

        // The forces change due to the configuration as determined by the jacobian

        // TODO: The finite difference is correct, need to re-create with analytic derivatives. look at frame velocity derivative stuff
        // pinocchio::computeForwardKinematicsDerivatives(pin_model_, *pin_data_, q, v, a);
        // matrixx_t dtf_dq = matrixx_t::Zero(GetVelDim(), GetVelDim());
        // matrix6x_t j1, j2;
        // j1 = matrixx_t::Zero(6, GetVelDim());
        // j2 = j1;
        // int idx = 0;
        // for (const auto& f : f_ext) {
        //     pinocchio::getFrameVelocityDerivatives(pin_model_, *pin_data_, GetFrameIdx(f.frame_name), pinocchio::LOCAL_WORLD_ALIGNED, j1, j2);
        //     dtf_dq.middleCols<3>(idx) = -j1.topRows<3>().transpose();
        //     idx += 3;
        // }

        // for (int i = 0; i < forces.size(); i++) {
        //     forces[i].linear() = f_ext[i].force_linear;
        //     forces[i].angular().setZero();
        // }
        // pinocchio::computeStaticTorqueDerivatives(pin_model_, *pin_data_, q, forces, dtf_dq);

        // std::cout << "analytic: \n" << dtf_dq << std::endl;

        // TODO: Move to codegen
        // Try with finite difference
        vectorx_t force_vec = vectorx_t::Zero(3*f_ext.size());
        int f_idx = 0;
        for (const auto& f : f_ext) {
            force_vec.segment<3>(f_idx) = f.force_linear;
            f_idx += 3;
        }

        matrixx_t fd = matrixx_t::Zero(GetVelDim(), GetVelDim());
        vectorx_t tau1 = dtau_df*force_vec;
        vectorx_t v_eps = vectorx_t::Zero(GetVelDim());
        vectorx_t q2, tau2;
        static constexpr double FD_DELTA = 1e-8;
        for (int i = 0; i < GetVelDim(); i++) {
            v_eps(i) += FD_DELTA;
            q2 = pinocchio::integrate(pin_model_, q, v_eps);
            tau2 = vectorx_t::Zero(GetVelDim());
            for (const auto& f : f_ext) {
                pinocchio::computeFrameJacobian(pin_model_, *pin_data_, q2, GetFrameIdx(f.frame_name), pinocchio::LOCAL_WORLD_ALIGNED, jacobian);
                tau2 += -jacobian.topRows<3>().transpose()*f.force_linear;
                jacobian.setZero();
            }

            fd.col(i) = (tau2 - tau1)/FD_DELTA;

            v_eps(i) -= FD_DELTA;
        }

        // std::cout << "fd: \n" << fd << std::endl;

        // matrixx_t df_dq = ExternalForcesDerivativeWrtConfiguration(q, f_ext);

        dtau_dq = dtau_dq + fd;

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

    vector3_t FullOrderRigidBody::QuaternionIntegrationRelative(const quat_t& qbar_kp1, const quat_t& qbar_k,
        const vector3_t& xi, const vector3_t& w, double dt) {
        return pinocchio::quaternion::log3(
            (qbar_kp1).inverse()
            *qbar_k*pinocchio::quaternion::exp3(xi)*pinocchio::quaternion::exp3(w*dt));
    }

    void FullOrderRigidBody::FrameVelDerivWrtConfiguration(const vectorx_t& q,
        const vectorx_t& v, const vectorx_t& a, const std::string& frame, matrix6x_t& jacobian,
        const pinocchio::ReferenceFrame& ref) {

        // --------- Finite Diff
        constexpr double FD_DELTA = 1e-8;
        jacobian.setZero();
        vectorx_t q_pert = q;
        vector3_t frame_vel = GetFrameState(frame, q, v, pinocchio::LOCAL_WORLD_ALIGNED).vel.linear();
        for (int i = 0; i < GetVelDim(); i++) {
            PerturbConfiguration(q_pert, FD_DELTA, i);
            vector3_t frame_pos_pert = GetFrameState(frame, q_pert, v, pinocchio::LOCAL_WORLD_ALIGNED).vel.linear();
            q_pert = q;

            jacobian.block(0, i, 3, 1) = (frame_pos_pert - frame_vel)/FD_DELTA;
        }
        // ---------



        // --------- Analytic
        // TODO: Consider swapping for auto diff
        // pinocchio::computeForwardKinematicsDerivatives(pin_model_, *pin_data_, q, v, a);
        //
        // matrix6x_t j2(6, GetVelDim());
        //
        // pinocchio::getFrameVelocityDerivatives(pin_model_, *pin_data_,
        //     GetFrameIdx(frame), ref, jacobian, j2);
        //
        // // TODO: The first 6 elements don't match because the jacobian is using local velocity vectors, I want perturbations in the global config
        // // So for now, we will use finite differencing on the first few elements
        // // TODO: Put back
        // vector3_t frame_vel = GetFrameState(frame, q, v, ref).vel.linear();
        //
        // static constexpr double FD_DELTA = 1e-8;
        // vectorx_t q_pert = q;
        // for (int i = 0; i < FLOATING_VEL; i++) {
        //     PerturbConfiguration(q_pert, FD_DELTA, i);
        //     vector3_t frame_pos_pert = GetFrameState(frame, q_pert, v, ref).vel.linear();
        //     q_pert = q;
        //
        //     for (int j = 0; j < frame_vel.size(); j++) {
        //         jacobian(j, i) = (frame_pos_pert(j) - frame_vel(j))/FD_DELTA;
        //     }
        // }
        // ---------
    }


    vectorx_t FullOrderRigidBody::InputsToTau(const vectorx_t& input) const {
        assert(input.size() == act_mat_.cols());
        return act_mat_*input;
    }

    pinocchio::container::aligned_vector<pinocchio::Force> FullOrderRigidBody::ConvertExternalForcesToPin(const vectorx_t& q,
            const std::vector<ExternalForce<double>>& f_ext) const {
        pinocchio::forwardKinematics(pin_model_, *pin_data_, q);
        pinocchio::framesForwardKinematics(pin_model_, *pin_data_, q);

        // Convert force to a pinocchio force
        pinocchio::container::aligned_vector<pinocchio::Force> forces(this->GetNumJoints(), pinocchio::Force::Zero());
        // TODO: Is this just data.oMi.act(data.f)? Like here: https://github.com/stack-of-tasks/pinocchio/blob/c989669e255715e2fa2504b3226664bf20de6fb5/unittest/rnea-derivatives.cpp#L143
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
            forces.at(jnt_idx).linear() += contact_force;

            // Calculate the angular (torque) forces
            forces.at(jnt_idx).angular() += translationContactToJoint.cross(contact_force);
        }
        return forces;
    }

    // TODO: Delete
    matrixx_t FullOrderRigidBody::ExternalForcesDerivativeWrtConfiguration(const vectorx_t& q, const std::vector<ExternalForce<double>>& f_ext) {
        // For now we will finite difference, but later we will use ad codegen
        // TODO: Use codegen
        const static double FD_DELTA = 1e-8;
        matrixx_t df_dq = matrixx_t::Zero(GetVelDim(), GetVelDim());
        vectorx_t v_eps =vectorx_t::Zero(GetVelDim());
        vectorx_t q2 = q;

        for (int i = 0; i < GetVelDim(); i++) {
            v_eps(i) += FD_DELTA;
            q2 = pinocchio::integrate(pin_model_, q, v_eps);
            v_eps(i) -= FD_DELTA;
        }

        return df_dq;


//         // sum of J(q) * df_dq
//         // J(q) are the frame jacobians for the joints
//         // df_dq is the derivative of ConvertExternalForcesToPin
//         pinocchio::forwardKinematics(pin_model_, *pin_data_, q);
//         pinocchio::framesForwardKinematics(pin_model_, *pin_data_, q);
//
// //        matrixx_t df_dq = matrixx_t::Zero(this->GetVelDim(), this->GetVelDim());
//         matrix6x_t df_dq = matrixx_t::Zero(6, this->GetVelDim());
//         for (const auto& f : f_ext) {
//             // Get the contact frame
//             const int frame_idx = this->GetFrameIdx(f.frame_name);
//             // Get the parent frame
//             const int jnt_idx = this->pin_model_.frames.at(frame_idx).parentJoint;
//             const vector3_t translationContactToJoint = pin_model_.frames.at(frame_idx).placement.translation();
//
//             // Get the contact Jacobian
// //            matrix6x_t J = matrix6x_t::Zero(6, this->GetVelDim());
// //            pinocchio::computeFrameJacobian(pin_model_, *pin_data_, q, frame_idx, pinocchio::LOCAL_WORLD_ALIGNED, J);
//
//             // Get the derivative of joint frame wrt q (i.e. a frame jacobian)
//             std::cout << "true joint name: " << this->pin_model_.names.at(jnt_idx) << std::endl;
//             int joint_frame_idx = this->GetFrameIdx(this->pin_model_.names.at(jnt_idx));
//             std::cout << "joint frame: " << this->pin_model_.frames.at(joint_frame_idx).name << std::endl;
//             std::cout << "torso frame: " << this->pin_model_.frames.at(GetFrameIdx("base")).name << std::endl;
//
//             matrix6x_t joint_frame_jacobian = matrix6x_t::Zero(6, this->GetVelDim());
//             pinocchio::computeFrameJacobian(pin_model_, *pin_data_, q, joint_frame_idx, pinocchio::LOCAL_WORLD_ALIGNED, joint_frame_jacobian);
//
//             matrix6x_t joint_frame_jacobian_world = matrix6x_t::Zero(6, this->GetVelDim());
//             pinocchio::computeFrameJacobian(pin_model_, *pin_data_, q, joint_frame_idx, pinocchio::WORLD, joint_frame_jacobian_world);
//
//             matrix6x_t torso_frame_jacobian_world = matrix6x_t::Zero(6, this->GetVelDim());
//             pinocchio::computeFrameJacobian(pin_model_, *pin_data_, q, GetFrameIdx("base"), pinocchio::LOCAL_WORLD_ALIGNED, torso_frame_jacobian_world);
//
//             std::cout << "Joint frame jacobian: \n" << joint_frame_jacobian << std::endl;
//             std::cout << "Joint frame jacobian wrt world: \n" << joint_frame_jacobian_world << std::endl;
//             std::cout << "Torso frame jacobian wrt world: \n" << torso_frame_jacobian_world << std::endl;
//
//             // Get the rotation from the world frame to the joint frame
//             const Eigen::Matrix3d rotationWorldToJoint = pin_data_->oMi[jnt_idx].rotation().transpose();
//
//             matrix6x_t dFdq = matrix6x_t::Zero(6, this->GetVelDim());
//             dFdq.topRows<3>() = joint_frame_jacobian_world.bottomRows<3>(); //joint_frame_jacobian_world.bottomRows<3>();
//             for (int i = 0; i < this->GetVelDim(); i++) {
//                 dFdq.block<3,1>(3, i) = translationContactToJoint.cross(joint_frame_jacobian_world.block<3,1>(3, i));
//             }
//
//             std::cout << "dFdq: \n" << dFdq << std::endl;
//
//             // TODO: I have the correct sparsity pattern and same general patterns, but the numbers are off
// //            df_dq = df_dq - joint_frame_jacobian.transpose()*dFdq;
//             df_dq = dFdq; // For now we just want to compare it against the function
//         }
//
//         return df_dq;
    }

    void FullOrderRigidBody::PerturbConfiguration(vectorx_t& q, double delta, int idx) {
        if (idx > GetVelDim()) {
            throw std::runtime_error("Invalid q perturbation index!");
        }

        if (idx < 3) {
            q(idx) += delta;
        } else if (idx < 6) {
            vector3_t q_pert = vector3_t::Zero();
            q_pert(idx - 3) += delta;
            q.array().segment<4>(3) = (static_cast<quat_t>(q.segment<4>(3)) * pinocchio::quaternion::exp3(q_pert)).coeffs();
        } else {
            q(idx + 1) += delta;
        }
    }

    vectorx_t FullOrderRigidBody::InverseKinematics(const vectorx_t& base_config, const std::vector<vector3_t>& positions, const std::vector<std::string>& frames,
        const vectorx_t& q_guess, bool use_floating_base) {
        // For each position I will need to update the tree's joints.
        // Note that I assume that each position is on an independent part of the tree.

        // TODO: Try to speed this up!

        if (base_config.size() != FLOATING_CONFIG) {
            throw std::runtime_error("[IK] Invalid base config size!");
        }

        if (frames.size() != positions.size()) {
            throw std::runtime_error("[IK] Number of positions and frames do not match!");
        }

        // Joint Bounds
        const vectorx_t lower_joint_lims = GetLowerConfigLimits();
        const vectorx_t upper_joint_lims = GetUpperConfigLimits();

        // Define constants for the optimization
        constexpr double THRESHOLD = 1e-4; //3
        constexpr int IT_MAX = 5e2;
        constexpr double DT = 1e-2;
        constexpr double DAMP = 1e-6;

        vectorx_t q = q_guess;
        q.head<FLOATING_CONFIG>() = base_config;

        matrix6x_t J = matrix6x_t::Zero(6, GetVelDim());
        matrix3x_t Jlin = matrix3x_t::Zero(3, GetVelDim());
        vectorx_t v = vectorx_t::Zero(GetVelDim());

        for (int ee = 0; ee < positions.size(); ee++) {
            const int frame_idx = GetFrameIdx(frames[ee]);
            v.setZero();

            // pinocchio::framesForwardKinematics(pin_model_, *pin_data_, q);
            // const vector3_t error = positions[ee] - pin_data_->oMf[frame_idx].translation();
            // std::cout << "\nq: " << q.transpose() << std::endl;
            // std::cout << "v: " << v.transpose() << std::endl;
            // std::cout << "Jlin:\n" << Jlin << std::endl;
            // std::cout << "current position: " << pin_data_->oMf[frame_idx].translation().transpose() << std::endl;
            // std::cout << "end effector: " << ee << ", iteration: " << 0 << ", error: " << error.transpose() << ", error norm: " << error.norm() << std::endl;

            for (int i = 0; i < IT_MAX; i++) {
                pinocchio::framesForwardKinematics(pin_model_, *pin_data_, q);
                const vector3_t error = -positions[ee] + pin_data_->oMf[frame_idx].translation();

                if (error.norm() <= THRESHOLD) {
                    break;
                } else if (i == IT_MAX - 1) {
                    std::cout << "[IK] Inverse Kinematics failed! Error: " << error.norm() << std::endl;
                    std::cout << "Using fixed floating base: " << !use_floating_base << std::endl;
                    std::cout << "q: " << q.transpose() << std::endl;
                    std::cout << "current position: " << pin_data_->oMf[frame_idx].translation().transpose() << std::endl;
                    std::cout << "target: " << positions[ee].transpose() << std::endl;
                    vector3_t error = -positions[ee] + pin_data_->oMf[frame_idx].translation();
                    std::cout << "end effector: " << ee << ", iteration: " << i << ", error: " << error.transpose() << ", error norm: " << error.norm() << std::endl;
                    std::cout << "frame: " << frames[ee] << std::endl << std::endl;
                }

                Jlin.setZero();
                J.setZero();
                pinocchio::computeFrameJacobian(pin_model_, *pin_data_, q, frame_idx, pinocchio::LOCAL_WORLD_ALIGNED, J);
                // Jlin = J.block(0, FLOATING_VEL, 3, GetVelDim() - FLOATING_VEL);
                Jlin = J.topRows<3>();
                if (!use_floating_base) {
                    Jlin.leftCols<FLOATING_VEL>().setZero();
                } else {
                    Jlin.leftCols<2>().setZero();
                }

                for (int j = 0; j < Jlin.cols(); j++) {
                    if (q(j) == upper_joint_lims(j) || q(j) == lower_joint_lims(j)) {
                        Jlin.col(j).setZero();
                    }
                }

                matrix3x_t JJt;
                JJt.noalias() = Jlin * Jlin.transpose();
                // TODO: Consider putting damping back in
                JJt.diagonal().array() += DAMP;

                v.noalias() = -Jlin.transpose() * JJt.ldlt().solve(error);
                if (!use_floating_base) {
                    v.head<FLOATING_VEL>().setZero();
                } else {
                    v.head<2>().setZero();
                }
                double alpha = 1;
                while (alpha >= DT) {
                    q = pinocchio::integrate(pin_model_, q, v*alpha);
                    pinocchio::framesForwardKinematics(pin_model_, *pin_data_, q);
                    q = pinocchio::integrate(pin_model_, q, -v*alpha);

                    const vector3_t error_ls = positions[ee] - pin_data_->oMf[frame_idx].translation();

                    if (error_ls.norm() < error.norm()) {
                        break;
                    }

                    alpha *= 0.5;
                }

                // if (alpha < DT) {
                //     // Not getting more decrease. Break the loop.
                //     break;
                // }

                // std::cout << "alpha: " << alpha << std::endl;

                q = pinocchio::integrate(pin_model_, q, v*alpha);

                // Enforce joint limits
                for (int i = 0; i < q.size(); i++) {
                    q(i) = std::min(std::max(q(i), lower_joint_lims(i)), upper_joint_lims(i));
                }

                // TODO: Remove after debugged
                // if (!(i%10)) {
                //     pinocchio::framesForwardKinematics(pin_model_, *pin_data_, q);
                //     std::cout << "q: " << q.transpose() << std::endl;
                //     std::cout << "v: " << v.transpose() << std::endl;
                //     std::cout << "Jlin:\n" << Jlin << std::endl;
                //     std::cout << "current position: " << pin_data_->oMf[frame_idx].translation().transpose() << std::endl;
                //     vector3_t error = -positions[ee] + pin_data_->oMf[frame_idx].translation();
                //     std::cout << "end effector: " << ee << ", iteration: " << i << ", error: " << error.transpose() << ", error norm: " << error.norm() << std::endl;
                // }
            }

            // std::cout << std::endl;
        }

        // TODO: Remove after debugged
        // pinocchio::framesForwardKinematics(pin_model_, *pin_data_, q);
        // for (int ee = 0; ee < positions.size(); ee++) {
        //     const int frame_idx = GetFrameIdx(frames[ee]);
        //     const vector3_t error = -positions[ee] + pin_data_->oMf[frame_idx].translation();
        //     std::cout << "end effector: " << ee << ", error norm: " << error.norm() << std::endl;
        //     std::cout << "resulting position: " << pin_data_->oMf[frame_idx].translation().transpose() << std::endl;
        // }

        return q;
    }

    // DEBUG ------
    pinocchio::Model FullOrderRigidBody::GetModel() const {
        return pin_model_;
    }
    // DEBUG ------


//    vectorx_t FullOrderRigidBody::TauToInputs(const vectorx_t& tau) const {
//        assert(tau == this->GetNumJoints());
//        return
//    }
} // torc::models
