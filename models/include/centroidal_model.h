//
// Created by gavin on 6/24/24.
//

#ifndef CENTROIDAL_MODEL_H
#define CENTROIDAL_MODEL_H

#include "pinocchio_model.h"

namespace torc::models {
    class CentroidalModel: public PinocchioModel {
    public:
        using vectorx_t = Eigen::VectorXd;
        using matrixx_t = Eigen::MatrixXd;

        CentroidalModel::CentroidalModel(const std::string &name,
                                         const std::filesystem::path &urdf,
                                         const std::vector<std::string>& contact_frames,
                                         const std::vector<std::string>& underactuated_joints);

        /**
         * @brief Compute the robot state derivative given a state and contact forces and set velocities
         * @param state The state of the robot (q and v, assumes q contains joint positions and v the current velocities)
         * @param input The set velocities of the joints and the contact forces. Dimension njoints + ncontacts*3
         * @return The derivative of the robot state (v and a, v contains CoM and joint velocities, a contains CoM accels)
         */
        RobotStateDerivative GetDynamics(const RobotState& state,
                                         const vectorx_t& input) override;

        void DynamicsDerivative(const RobotState& state, const vectorx_t& input,
                                        matrixx_t& A, matrixx_t& b) override;

        RobotState GetImpulseDynamics(const RobotState& state, const vectorx_t& input,
                          const RobotContactInfo& contact_info);

        void CentroidalModel::RegisterUnactuatedJoints(const std::vector<std::string>& underactuated_joints);
        vectorx_t CentroidalModel::InputsToTau(const vectorx_t& input) const;

    private:
        std::vector<pinocchio::FrameIndex> contact_frames_idxs_;
        std::vector<int> unactuated_joint_idxs_;
    };
}

#endif //CENTROIDAL_MODEL_H
