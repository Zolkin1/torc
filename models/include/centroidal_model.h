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
         * @param input The set velocities of the joints and the contact forces. Dimension ncontacts*3 + nactuated, layout
         * is [fc_1, \ldots, fc_nc, v_1, \ldots, v_a]
         * @return The derivative of the state wrt time, v=joint position velocities, a=center of mass momentum change
         */
        RobotStateDerivative GetDynamics(const RobotState& state,
                                         const vectorx_t& input) override;

        /**
         * @brief Linearizes the dynamics into the form $\partial_x xdot = A, \partial_u xdot = B$
         *
         * Strategy: we consider the velocity and acceleration separately.
         *
         * dv (joints): if the joint is unactuated, $dv_i=dx_i$. Otherwise, $dv_i=du_i$
         * dv (CoM):
         * da (CoM): a =
         *
         * @param state The state of the robot (q and v, assumes q contains joint positions and v the current velocities)
         * @param input The set velocities of the joints and the contact forces
         * @param A
         * @param B
         */
        void DynamicsDerivative(const RobotState& state,
                                const vectorx_t& input,
                                matrixx_t& A,
                                matrixx_t& B) override;

        void CentroidalModel::RegisterUnactuatedJoints(const std::vector<std::string>& underactuated_joints);

    private:
        std::vector<pinocchio::FrameIndex> contact_frames_idxs_;
        std::vector<int> unactuated_joint_idxs_;
    };
}

#endif //CENTROIDAL_MODEL_H
