#include "centroidal_model.h"

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/centroidal.hpp>

namespace torc::models {
    CentroidalModel::CentroidalModel(const std::string &name,
                                     const std::filesystem::path &urdf,
                                     const std::vector<std::string>& contact_frames,
                                     const std::vector<std::string>& underactuated_joints)
        :PinocchioModel(name, urdf) {
        for (const auto& frame_name : contact_frames) {
            contact_frames_idxs_.emplace_back(pin_model_.getFrameId(frame_name));
        }
        RegisterUnactuatedJoints(underactuated_joints);
    }

    RobotStateDerivative CentroidalModel::GetDynamics(const RobotState &state,
                                                      const vectorx_t& input)
    {
        state.AssertConsistentDimension();
        constexpr int V_DIM = 3;
        constexpr int ROOT_JOINT_IDX = 1;

        vectorx_t v_new = state.v;
        int input_idx = 0;
        for (int i=0; i<v_new.size(); i++) {
            if (std::find(unactuated_joint_idxs_.begin(), unactuated_joint_idxs_.end(), i) == unactuated_joint_idxs_.end()) {
                // joint is actuated
                v_new(i) = input(input_idx++);
            }
        }

        vectorx_t contact_forces = input.bottomRows(contact_frames_idxs_.size()*V_DIM); // assuming no contact torques
        forwardKinematics(pin_model_, *pin_data_, state.q);
        vectorx_t force_sum = {0, 0, -9.81};
        vectorx_t torque_sum = vectorx_t::Zero(V_DIM);
        pinocchio::SE3 root_pos = pin_data_->oMi[ROOT_JOINT_IDX];

        for (int i=0; i<contact_forces.rows(); i+=V_DIM) {
            // iterate through the forces and compute their effect on the CoM
            const vectorx_t force = contact_forces.middleRows(i, V_DIM);
            force_sum += force;
            pinocchio::SE3 contact_frame_pos = pin_data_->oMi[contact_frames_idxs_.at(i)];
            vectorx_t r = contact_frame_pos.translation() - root_pos.translation();
            torque_sum += r.cross3(force);
        }

        vectorx_t dh_CoM(force_sum.size() + torque_sum.size());
        dh_CoM << force_sum / mass_, torque_sum;

        // Compute the kinematics and dynamics
        pinocchio::crba(pin_model_, *pin_data_, state.q);
        pin_data_->Ag;      // the A_j
        pinocchio::computeCentroidalMomentum(pin_model_, *pin_data_);
        const vectorx_t h_CoM = pin_data_->hg.toVector();
        vectorx_t state_v_new(v_new.size() + h_CoM.size());
        state_v_new << h_CoM, v_new;

        return {state_v_new, dh_CoM};
        // return: new velocities (COM and all joints), accel (COM only)
    }


    void CentroidalModel::RegisterUnactuatedJoints(const std::vector<std::string>& underactuated_joints) {
        assert(pin_model_.idx_vs.at(1) == 0);
        assert(pin_model_.nvs.at(1) == FLOATING_VEL);

        int num_actuators = pin_model_.nv;
        const int num_joints = pin_model_.njoints;

        std::vector<int> unact_joint_idx;

        unactuated_joint_idxs_.push_back(0);   // Universe joint is never actuated

        // Get the number of actuators
        for (const std::string& joint_name : underactuated_joints) {
            for (int i = 0; i < num_joints; i++) {
                if (joint_name == pin_model_.names.at(i)) {
                    num_actuators -= pin_model_.joints.at(i).nv();
                    unactuated_joint_idxs_.push_back(i);
                    break;
                }
            }
        }
    }
}