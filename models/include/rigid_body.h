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

    class RigidBody : public PinocchioModel{
    public:

        RigidBody(std::string name, std::filesystem::path urdf);

        // @note These are not actually const functions as we modify the pin_data struct
        [[nodiscard]] RobotStateDerivative GetDynamics(const RobotState& state, const vectorx_t& input) const override;

        [[nodiscard]] RobotStateDerivative GetDynamics(const RobotState& state, const vectorx_t& input,
                                                    const RobotContactInfo& contact_info) const;

        RobotState GetImpulseDynamics(const RobotState& state, const vectorx_t& input,
                                      const RobotContactInfo& contact_info) const;

        void DynamicsDerivative(const RobotState& state, const vectorx_t& input,
                                matrixx_t& A, matrixx_t& B) const override;


        void DynamicsDerivative(const RobotState& state, const vectorx_t& input, const RobotContactInfo& contacts,
                                matrixx_t& A, matrixx_t& B) const;


        void ImpulseDerivative(const RobotContactInfo& contact_info,
                               matrixx_t& A, matrixx_t& B) const;

    protected:

    private:

    };
} // torc::models


#endif //TORC_RIGID_BODY_H
