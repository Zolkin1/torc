//
// Created by gavin on 6/24/24.
//

#ifndef CENTROIDAL_MODEL_H
#define CENTROIDAL_MODEL_H

#include "pinocchio_model.h"
#include <unordered_set>

namespace torc::models {
    /**
     * @brief Implementation of the centroidal model (Sleimann et. al., 2021). In this model, the state of the robot
     * consists purely of the joint positions and the generalized CoM momenta. It is also assumed that the actuated
     * joints can be given any speed at any time without acceleration.
     *
     * q = [q_base, q_joints], v = [h_CoM], a = [dh_CoM]
     */
    
    class Centroid: public PinocchioModel {

    public:
        using vectorx_t = Eigen::VectorXd;
        using matrixx_t = Eigen::MatrixXd;
        using vec3 = Eigen::Vector3d;

        Centroid(const std::string &name,
                        const std::filesystem::path &urdf,
                        const std::vector<std::string>& contact_frames,
                        const std::vector<std::string>& unactuated_joints);

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
         * @param state The state of the robot (q and v, assumes q contains joint positions and v the current velocities)
         * @param input The set velocities of the joints and the contact forces
         * @param A Derivative of robot state's time derivative with respect to the robot state
         * @param B Derivative of the robot state's time derivative with respect to the contact forces and set speeds.
         */
        void DynamicsDerivative(const RobotState& state,
                                const vectorx_t& input,
                                matrixx_t& A,
                                matrixx_t& B) override;

        RobotState GetRandomState() const;
        vectorx_t GetRandomInput() const;

        int GetDerivativeDim() const;
        int GetInputDim() const;


    private:
        static constexpr int FORCE_DIM = 3;
        static constexpr int LINEAR_DIM = 3;
        static constexpr int ANGULAR_DIM = 3;
        static constexpr int BASE_DOF = LINEAR_DIM + ANGULAR_DIM;
        static constexpr int COM_DOF = LINEAR_DIM + ANGULAR_DIM;

        std::vector<pinocchio::FrameIndex> contact_frames_idxs_; // indicies of contact frames
        std::unordered_set<int> unactuated_joint_idxs_;          // indicies of unactuated joints
        const int n_contacts_;                                // number of contacts
        const int n_actuated_;                                // number of actuated joints

        /**
         * @brief Extracts the contact forces in the input into a vector of 3-vectors.
         * @param input The input, in the form [f_c1, ..., f_cn, v_j]
         * @return The contact forces in the input
         */
        [[nodiscard]] std::vector<vec3> GetForcesFromInput(const vectorx_t& input) const;

        // /**
        //  * @brief Computes the actuation map, which maps the set velocities to the joint velocities, with unactauated joints
        //  * set to 0 velocity. It is effectively an identity matric with zero rows inserted at the indicies of the
        //  * unactuated joints.
        //  * @return The actuation map
        //  */
        // [[nodiscard]] matrixx_t GetActuationMap() const;

        /**
         * @brief Updates the joint velocities of the robot using an input, assuming that all joints can be infinitely
         * accelerated. The resulting vector is that all actuated joints' velocities are set to their respective input
         * values, while the unactuated joints remain constant.
         * @param state The state of the robot [q, v]
         * @param input The input to the robot [F, v_j]
         * @return The new joint velocities
         */
        [[nodiscard]] vectorx_t UpdateJointVelocities(const RobotState& state, const vectorx_t& input) const;

        vectorx_t InputsToTau(const vectorx_t &input) const override {return vectorx_t::Zero(0);}

    };
}

#endif //CENTROIDAL_MODEL_H
