#include "centroidal_model.h"

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/centroidal.hpp>

namespace torc::models {
    // vector of strings -> vector of frames to consider for contact
    // then assume everything is in that order
    CentroidalModel::CentroidalModel(const std::string &name,
                                     const std::filesystem::path &urdf,
                                     const std::vector<std::string>& contact_frames)
        :PinocchioModel(name, urdf) {
        for (const auto& frame_name : contact_frames) {
            contact_frames_idxs_.emplace_back(pin_model_.getFrameId(frame_name));
        }
    }

    // for now, keep inputs in the same vector
    RobotStateDerivative CentroidalModel::GetDynamics(const RobotState &state,
                                                      const vectorx_t& input)
    // forward kinematics
    // get location of the frame
    {
        state.AssertConsistentDimension();
        constexpr int V_DIM = 3;
        constexpr int ROOT_JOINT_IDX = 1;
        vectorx_t v_new = state.v;
        v_new.block(V_DIM, 0, pin_model_.njoints*V_DIM, 1) = input.block(0, 0, pin_model_.njoints*V_DIM, 1);
        vectorx_t contact_forces = input.bottomRows(contact_frames_idxs_.size()*V_DIM); // assuming no contact torques
        forwardKinematics(pin_model_, *pin_data_, state.q);
        vectorx_t force_sum = {0, 0, -9.81};
        vectorx_t torque_sum = vectorx_t::Zero(V_DIM);
        pinocchio::SE3 root_pos = pin_data_->oMi[ROOT_JOINT_IDX];

        for (int i=0; i<contact_forces.rows(); i++) {
            const vectorx_t force = contact_forces.middleRows(i, V_DIM);
            force_sum += force;
            pinocchio::SE3 contact_frame_pos = pin_data_->oMi[contact_frames_idxs_.at(i)];
            vectorx_t r = contact_frame_pos.translation() - root_pos.translation();
            torque_sum += r.cross3(force);
        }

        // Compute the kinematics and dynamics
        pinocchio::crba(pin_model_, *pin_data_, state.q);
        pin_data_->Ag;      // the A_j
        pinocchio::computeCentroidalMomentum(pin_model_, *pin_data_);
        pin_data_->hg;
        // pinocchio::computeCentroidal

        // Now access the centroidal mass matrix
        pinocchio::container::aligned_vector<pinocchio::Force> aligned_forces;
        // for (auto force : forces) {
            // aligned_forces.push_back(force);
        // }
        // aba(pin_model_, *pin_data_, state.q, v_new, vectorx_t::Zero(state.v_dim()), aligned_forces);
        // no accel in joints
        return {v_new, pin_data_->ddq};
    }
}