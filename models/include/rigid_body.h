//
// Created by zolkin on 5/23/24.
//

#ifndef TORC_RIGID_BODY_H
#define TORC_RIGID_BODY_H

#include "pinocchio_model.h"
#include "robot_state_types.h"
#include "robot_contact_info.h"

namespace torc::models {
    using vectorx_t = Eigen::VectorXd;
    using matrixx_t = Eigen::MatrixXd;

    class RigidBody : public PinocchioModel {
    public:

        RigidBody(const std::string& name, const std::filesystem::path& urdf);

        RigidBody(const std::string& name, const std::filesystem::path& urdf,
                  const std::vector<std::string>& underactuated_joints);

        RigidBody(const RigidBody& other);

        // @note These are not actually const functions as we modify the pin_data struct
        [[nodiscard]] RobotStateDerivative GetDynamics(const RobotState& state, const vectorx_t& input) override;

        [[nodiscard]] RobotStateDerivative GetDynamics(const RobotState& state, const vectorx_t& input,
                                                    const RobotContactInfo& contact_info) const;

        RobotState GetImpulseDynamics(const RobotState& state, const vectorx_t& input,
                                      const RobotContactInfo& contact_info);

        // TODO: Linearization function that calculates the FD and derivatives so we don't have redundant calls

        void DynamicsDerivative(const RobotState& state, const vectorx_t& input,
                                matrixx_t& A, matrixx_t& B) override;


        void DynamicsDerivative(const RobotState& state, const vectorx_t& input, const RobotContactInfo& contacts,
                                matrixx_t& A, matrixx_t& B);


        void ImpulseDerivative(const RobotState& state, const vectorx_t& input,
                               const RobotContactInfo& contact_info,
                               matrixx_t& A, matrixx_t& B);

        /**
         * Takes the torques on the actuated coordinates and maps to a vector of
         * dimension model.nv with zeros on underacutated joints
         * @param input
         * @return full input vector
         */
        [[nodiscard]] vectorx_t InputsToTau(const vectorx_t& input) const override;
    protected:
        void CreateActuationMatrix(const std::vector<std::string>& underactuated_joints);

        matrixx_t act_mat_;

        std::unique_ptr<pinocchio::Data> contact_data_;
    private:

    };
} // torc::models


#endif //TORC_RIGID_BODY_H
