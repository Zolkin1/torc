#include "centroidal_model.h"

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/centroidal.hpp>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/algorithm/model.hpp>

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

        // update all joint velocities, except base joint, compute dq_joints
        vectorx_t dq_joints = state.v;
        size_t dq_joint_input_idx = contact_frames_idxs_.size()*V_DIM; // skip contact forces segment
        for (int i=2; i<dq_joints.size(); i++) {        // skip world joint and base joint
            if (std::find(unactuated_joint_idxs_.begin(), unactuated_joint_idxs_.end(), i) == unactuated_joint_idxs_.end()) {
                // joint is actuated
                dq_joints(i) = input(dq_joint_input_idx++);
            }
        }

        // compute CoM position, CMM, all frame positions, rotational inertia, and base joint velocity
        pinocchio::computeCentroidalMomentum(pin_model_, *pin_data_, state.q, dq_joints);
        pinocchio::computeCentroidalMap(pin_model_, *pin_data_, state.q);
        pinocchio::forwardKinematics(pin_model_, *pin_data_, state.q);
        pinocchio::crba(pin_model_, *pin_data_, state.q);

        // determine CoM momentum
        // const Eigen::Matrix3d com_inertia = pin_data_->Ycrb[0].inertia();   // rotational inertia
        // const vectorx_t v_com = h_com.segment(0, 3) / mass_;
        // const vectorx_t omega_com = com_inertia.inverse() * h_com.segment(3, 6);    // avoid inverting a 6x6
        // vectorx_t dq_com(6);
        // dq_com << v_com, omega_com;

        // determine dh_com by aggregating all the forces
        const vectorx_t com_pos = pin_data_->com[0];
        const vectorx_t contact_forces = input.topRows(contact_frames_idxs_.size()*V_DIM); // assuming no contact torques
        vectorx_t force_sum = {0, 0, -9.81*mass_};        // gravity is always present
        vectorx_t torque_sum = vectorx_t::Zero(V_DIM);          // gravity provides no torque
        // iterate through the forces
        for (int i=0; i<contact_forces.rows(); i+=V_DIM) {
            const vectorx_t force = contact_forces.middleRows(i, V_DIM);
            force_sum += force;
            pinocchio::SE3 contact_frame_pos = pin_data_->oMi[contact_frames_idxs_.at(i)];
            vectorx_t r = contact_frame_pos.translation() - com_pos;    // assume that all the forces are expressed in the CoM frame
            torque_sum += r.cross3(force);
        }
        vectorx_t dh_com(force_sum.size() + torque_sum.size());
        dh_com << force_sum, torque_sum;

        // compute dq_base
        const matrixx_t cmm = pin_data_->Ag;
        const matrixx_t A_base = cmm.leftCols(6);
        const matrixx_t A_joints = cmm.rightCols(cmm.cols() - 6);
        const vectorx_t h_com = pin_data_->hg.toVector();
        const vectorx_t dq_base = A_base.inverse() * (h_com - A_joints * dq_joints);

        // Join together dq_b and dq_j
        vectorx_t dq(dq_base.size() + dq_joints.size());
        dq << dq_base, dq_joints;

        // the paper defines the state vector x as [hcom, qb, qj], so its derivative is [dhcom, dqb, dqj]
        // within our paradigm, dhcom is the acceleration, and the dq's are the velocities
        return {dq, dh_com};
    }

    void CentroidalModel::DynamicsDerivative(const RobotState &state,
                                             const vectorx_t &input,
                                             matrixx_t &A,
                                             matrixx_t &B) {

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
