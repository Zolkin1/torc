#include "centroidal_model.h"

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hxx>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/model.hpp>

namespace torc::models {
    Centroid::Centroid(const std::string &name,
                                     const std::filesystem::path &urdf,
                                     const std::vector<std::string>& contact_frames,
                                     const std::vector<std::string>& unactuated_joints)
        :PinocchioModel(name, urdf),
        n_contacts_(contact_frames.size()),
        n_actuated_(pin_model_.nv - BASE_DOF - unactuated_joints.size()){
        for (const auto& frame_name : contact_frames) {
            const size_t frame_id = pin_model_.getFrameId(frame_name);
            if (frame_id == -1) {
                throw std::runtime_error("Frame does not exist!");
            }
            contact_frames_idxs_.emplace_back(frame_id);
        }

        unactuated_joint_idxs_.emplace(0);   // Universe joint is never actuated
        unactuated_joint_idxs_.emplace(1);   // Base joint is never actuated

        for (const std::string& joint_name : unactuated_joints) {
            for (int i = 0; i < pin_model_.njoints; i++) {
                if (joint_name == pin_model_.names.at(i)) {
                    unactuated_joint_idxs_.emplace(i);
                    break;
                }
            }
        }
    }

    RobotStateDerivative Centroid::GetDynamics(const RobotState &state,
                                                      const vectorx_t& input)
    {
        // update all joint velocities, except base joint, compute dq_joints
        const vectorx_t dqj = UpdateJointVelocities(state, input);
        const std::vector<vec3> forces = GetForcesFromInput(input);    // TODO change to vec3

        pinocchio::crba(pin_model_, *pin_data_, state.q);
        pinocchio::computeCentroidalMap(pin_model_, *pin_data_, state.q);
        const matrixx_t &cmm = pin_data_->Ag;
        const matrixx_t A_b = cmm.leftCols(BASE_DOF);
        const matrixx_t A_j = cmm.rightCols(cmm.cols() - BASE_DOF);
        const vectorx_t dqb = A_b.inverse() * (state.v - A_j * dqj);

        // Join together dqb and dqj
        vectorx_t dq(dqb.size() + dqj.size());
        dq << dqb, dqj;

        // compute CoM position, CMM, all frame positions, rotational inertia
        pinocchio::computeCentroidalMomentum(pin_model_, *pin_data_, state.q, dq); // need to add dqb
        pinocchio::forwardKinematics(pin_model_, *pin_data_, state.q);

        // determine dh_com by aggregating all the forces
        const vectorx_t com_pos = pin_data_->com[0];
        vec3 force_sum = {0, 0, -9.81*mass_};        // gravity is always present
        vec3 torque_sum = vec3::Zero();                     // gravity provides no torque
        // iterate through the forces
        for (int i=0; i<n_contacts_; i++) {
            // assume that all the forces are expressed in the CoM frame
            const vectorx_t& force = forces.at(i);
            force_sum += force;
            vec3 r = pin_data_->oMi[contact_frames_idxs_.at(i)].translation() - com_pos;
            torque_sum += r.cross(vec3(force));
        }
        vectorx_t dh_com(force_sum.size() + torque_sum.size());
        dh_com << force_sum, torque_sum;

        return {dq, dh_com};
    }

    void Centroid::DynamicsDerivative(const RobotState &state,
                                             const vectorx_t &input,
                                             matrixx_t &A,
                                             matrixx_t &B) {

        const vectorx_t dqj = UpdateJointVelocities(state, input);
        const std::vector<vec3> contact_forces = GetForcesFromInput(input);


        forwardKinematics(pin_model_, *pin_data_, state.q);
        centerOfMass(pin_model_, *pin_data_, state.q, true);
        const matrixx_t A_g = computeCentroidalMap(pin_model_, *pin_data_, state.q);
        const matrixx_t A_b_inv = A_g.leftCols(BASE_DOF).inverse();
        const matrixx_t A_j = A_g.rightCols(pin_data_->Ag.cols() - BASE_DOF);
        const vectorx_t dqb = A_b_inv * (state.v - A_j * dqj);
        const matrixx_t &Jcom = pin_data_->Jcom;
        const vectorx_t com_pos = pin_data_->com[0];
        vectorx_t dq(dqj.size() + dqb.size());
        dq << dqb, dqj;

        computeCentroidalMomentum(pin_model_, *pin_data_, state.q, dq);

        // calculate ddx_dx (A)
        A = matrixx_t::Zero(COM_DOF + pin_model_.nv, COM_DOF + pin_model_.nv);
        A.block(COM_DOF, 0, COM_DOF, BASE_DOF) = A_b_inv;

        matrixx_t dhtau_dq(ANGULAR_DIM, pin_model_.nv);
        pinocchio::impl::computeJointJacobians(pin_model_, *pin_data_, state.q);
        // allocate outside of the loop to prevent reinitialization
        matrixx_t J_frame = matrixx_t::Zero(COM_DOF, pin_model_.nv);
        // iterate through all the contact forces so we don't have to recalculate frame jacobians every time
        for (int i=0; i<n_contacts_; i++) {
            const vectorx_t& force = contact_forces.at(i);
            J_frame = getFrameJacobian(pin_model_, *pin_data_, contact_frames_idxs_.at(i), pinocchio::WORLD);
            const matrixx_t dr_dq = J_frame - Jcom;
            // calculate the partial derivatives' effect on the angular momentum
            for (int j=0; j<pin_model_.nv; j++) {
                dhtau_dq.col(j) += vec3(dr_dq.col(j)).cross(vec3(force));
            }
            A.block(LINEAR_DIM, COM_DOF, ANGULAR_DIM, pin_model_.nv) = dhtau_dq;
        }

        // calculate ddx_du (B)
        B = matrixx_t::Zero(COM_DOF + pin_model_.nv, input.size());
        B.bottomRightCorner(n_actuated_, n_actuated_) = matrixx_t::Identity(n_actuated_, n_actuated_);
        B.block(COM_DOF, n_contacts_*FORCE_DIM, BASE_DOF, n_actuated_) = -A_b_inv * A_j;
        // iterate through the forces
        for (int i=0; i<n_contacts_; i++) {
            const matrixx_t iden3 = matrixx_t::Identity(LINEAR_DIM, FORCE_DIM);
            B.block(0, i*FORCE_DIM, LINEAR_DIM, FORCE_DIM) = iden3;  // linear
            vec3 r = pin_data_->oMi[contact_frames_idxs_.at(i)].translation() - com_pos;   // rotational
            matrixx_t dhtau_dfi = matrixx_t::Zero(ANGULAR_DIM, FORCE_DIM);
            for (int j=0; j<FORCE_DIM; j++) {
                dhtau_dfi.col(j) = r.cross(vec3(iden3.col(j)));
            }
            B.block(LINEAR_DIM, i*FORCE_DIM, ANGULAR_DIM, FORCE_DIM) = dhtau_dfi;
        }
    }

    RobotState Centroid::GetRandomState() const {
        vectorx_t rand_q(pin_model_.nq);
        rand_q << vectorx_t::Random(pin_model_.nq-4), Eigen::Quaterniond::UnitRandom().coeffs();
        vectorx_t rand_v = vectorx_t::Random(COM_DOF);
        return {rand_q, rand_v};
    }

    vectorx_t Centroid::GetRandomInput() const {
        return vectorx_t::Random(GetInputDim());
    }

    int Centroid::GetDerivativeDim() const {
        return pin_model_.nv + COM_DOF;
    }

    int Centroid::GetInputDim() const {
        return n_contacts_ * FORCE_DIM + n_actuated_;
    }


    std::vector<Centroid::vec3> Centroid::GetForcesFromInput(const vectorx_t &input) const {
        const vectorx_t contact_forces = input.topRows(n_contacts_ * FORCE_DIM); // assuming no contact torques
        std::vector<vec3> forces = {};
        for (int i=0; i<n_contacts_; i++) {
            forces.emplace_back(input.segment(i*FORCE_DIM, FORCE_DIM));
        }
        return forces;
    }


    // matrixx_t Centroid::GetActuationMap() const {
    //     matrixx_t actuation_map = matrixx_t::Zero(pin_model_.nv-6, static_cast<long>(n_actuated_));
    //     int input_idx = 0;
    //     for (int joint_idx=6; joint_idx<pin_model_.nv; joint_idx++) {
    //         if (unactuated_joint_idxs_.count(joint_idx) == 0) {
    //             // joint is actuated
    //             actuation_map(joint_idx-6, input_idx++) = 1;
    //         }
    //     }
    //     return actuation_map;
    // }


    vectorx_t Centroid::UpdateJointVelocities(const RobotState &state,
                                                     const vectorx_t &input) const {
        // update all joint velocities, except base and world joint, compute dqj
        // vectorx_t dqj = state.v;
        // size_t dqj_input_idx = n_contacts_*3;   // skip contact forces segment
        // for (int i=2; i<dqj.size(); i++) {                      // skip world joint and base joint
        //     if (unactuated_joint_idxs_.count(i) == 0) {
        //         // joint is actuated
        //         dqj(i) = input(static_cast<long>(dqj_input_idx));
        //         ++dqj_input_idx;
        //     }
        // }
        // return dqj;
        return input.segment(n_contacts_*FORCE_DIM, n_actuated_);
    }
}
