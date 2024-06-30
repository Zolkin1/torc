#include "centroidal_model.h"

#include <pinocchio/fwd.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/algorithm/jacobian.hxx>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/model.hpp>

namespace torc::models {
    CentroidalModel::CentroidalModel(const std::string &name,
                                     const std::filesystem::path &urdf,
                                     const std::vector<std::string>& contact_frames,
                                     const std::vector<std::string>& unactuated_joints)
        :PinocchioModel(name, urdf), n_contacts_(contact_frames.size()), n_actuated_(pin_model_.nv-6-unactuated_joints.size()){
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

    RobotStateDerivative CentroidalModel::GetDynamics(const RobotState &state,
                                                      const vectorx_t& input)
    {
        // update all joint velocities, except base joint, compute dq_joints
        const vectorx_t dqj = UpdateJointVelocities(state, input);
        const std::vector<vectorx_t> forces = GetForcesFromInput(input);

        // compute CoM position, CMM, all frame positions, rotational inertia
        pinocchio::computeCentroidalMomentum(pin_model_, *pin_data_, state.q, dqj);
        pinocchio::computeCentroidalMap(pin_model_, *pin_data_, state.q);
        pinocchio::forwardKinematics(pin_model_, *pin_data_, state.q);
        pinocchio::crba(pin_model_, *pin_data_, state.q);

        // determine dh_com by aggregating all the forces
        const vectorx_t com_pos = pin_data_->com[0];
        const vectorx_t contact_forces = input.topRows(n_contacts_*3); // assuming no contact torques
        vec3 force_sum = {0, 0, -9.81*mass_};        // gravity is always present
        vec3 torque_sum = vec3::Zero();                     // gravity provides no torque
        // iterate through the forces
        for (int i=0; i<contact_forces.rows(); i+=3) {
            // assume that all the forces are expressed in the CoM frame
            const vectorx_t& force = forces.at(i);
            force_sum += force;
            vec3 r = pin_data_->oMi[contact_frames_idxs_.at(i)].translation() - com_pos;
            torque_sum += r.cross(vec3(force));
        }
        vectorx_t dh_com(force_sum.size() + torque_sum.size());
        dh_com << force_sum, torque_sum;

        // compute dqb
        const matrixx_t cmm = pin_data_->Ag;
        const matrixx_t A_b = cmm.leftCols(6);
        const matrixx_t A_j = cmm.rightCols(cmm.cols() - 6);
        const vectorx_t h_com = pin_data_->hg.toVector();
        const vectorx_t dqb = A_b.inverse() * (h_com - A_j * dqj);

        // Join together dqb and dqj
        vectorx_t dq(dqb.size() + dqj.size());
        dq << dqb, dqj;

        return {dq, dh_com};
    }

    void CentroidalModel::DynamicsDerivative(const RobotState &state,
                                             const vectorx_t &input,
                                             matrixx_t &A,
                                             matrixx_t &B) {

        const vectorx_t dq_joints = UpdateJointVelocities(state, input);
        const std::vector<vectorx_t> contact_forces = GetForcesFromInput(input);


        pinocchio::forwardKinematics(pin_model_, *pin_data_, state.q);
        pinocchio::centerOfMass(pin_model_, *pin_data_, state.q, true);
        pinocchio::computeCentroidalMomentum(pin_model_, *pin_data_, state.q, dq_joints);
        pinocchio::computeCentroidalMap(pin_model_, *pin_data_, state.q);

        const matrixx_t Jcom = pin_data_->Jcom;
        const vectorx_t com_pos = pin_data_->com[0];
        const matrixx_t A_b = pin_data_->Ag.leftCols(6);
        const matrixx_t A_j = pin_data_->Ag.rightCols(pin_data_->Ag.cols() - 6);

        // calculate ddx_dx (A)
        A = matrixx_t::Zero(6 + pin_model_.nq, 6 + pin_model_.nq);
        A.block(6, 0, 6, 6) = A_b.inverse();

        matrixx_t dhtau_dq(3, pin_model_.nq);
        pinocchio::impl::computeJointJacobians(pin_model_, *pin_data_, state.q);
        // allocate outside of the loop to prevent reinitialization
        matrixx_t J_frame = matrixx_t::Zero(6, pin_model_.nq);
        // iterate through all the contact forces so we don't have to recalculate frame jacobians every time
        for (int i=0; i<n_contacts_; i++) {
            const vectorx_t& force = contact_forces.at(i);
            J_frame = pinocchio::getFrameJacobian(pin_model_, *pin_data_, contact_frames_idxs_.at(i), pinocchio::WORLD);
            const matrixx_t dr_dq = J_frame - Jcom;
            // calculate the partial derivatives' effect on the angular momentum
            for (int j=0; j<pin_model_.nq; j++) {
                dhtau_dq.col(j) += vec3(dr_dq.col(j)).cross(vec3(force));
            }
            A.block(3, 6, 3, pin_model_.nq) = dhtau_dq;
        }

        // calculate ddx_du (B)
        B = matrixx_t::Zero(6 + pin_model_.nq, input.size());
        B.bottomRightCorner(pin_model_.nq-6, n_actuated_) = GetActuationMap();
        B.block(6, static_cast<long>(n_contacts_*3), 3, pin_model_.nq) = -A_b.inverse() * A_j;
        // iterate through the forces
        for (int i=0; i<n_contacts_; i++) {
            const matrixx_t iden3 = matrixx_t::Identity(3, 3);
            B.block(0, i*3, 3, 3) = iden3;  // linear
            vec3 r = pin_data_->oMi[contact_frames_idxs_.at(i)].translation() - com_pos;   // rotational
            matrixx_t dhtau_dfi = matrixx_t::Zero(3, 3);
            for (int j=0; j<3; j++) {
                dhtau_dfi.col(j) = r.cross(vec3(iden3.col(j)));
            }
            B.block(3, i*3, 3, 3) = dhtau_dfi;
        }
    }


    std::vector<vectorx_t> CentroidalModel::GetForcesFromInput(const vectorx_t &input) const {
        const vectorx_t contact_forces = input.topRows(n_contacts_ * 3); // assuming no contact torques
        std::vector<vectorx_t> forces = {};
        for (int i=0; i<n_contacts_; i++) {
            forces.emplace_back(input.segment(i*3, 3));
        }
        return forces;
    }


    matrixx_t CentroidalModel::GetActuationMap() const {
        matrixx_t actuation_map = matrixx_t::Zero(pin_model_.nq-6, static_cast<long>(n_actuated_));
        int input_idx = 0;
        for (int joint_idx=6; joint_idx<pin_model_.nq; joint_idx++) {
            if (unactuated_joint_idxs_.count(joint_idx) == 0) {
                // joint is actuated
                actuation_map(joint_idx-6, input_idx++) = 1;
            }
        }
        return actuation_map;
    }


    vectorx_t CentroidalModel::UpdateJointVelocities(const RobotState &state,
                                                     const vectorx_t &input) const {
        // update all joint velocities, except base and world joint, compute dqj
        vectorx_t dqj = state.v;
        size_t dqj_input_idx = n_contacts_*3;   // skip contact forces segment
        for (int i=2; i<dqj.size(); i++) {                      // skip world joint and base joint
            if (unactuated_joint_idxs_.count(i) == 0) {
                // joint is actuated
                dqj(i) = input(static_cast<long>(dqj_input_idx));
                ++dqj_input_idx;
            }
        }
        return dqj;
    }
}
