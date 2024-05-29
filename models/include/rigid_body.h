//
// Created by zolkin on 5/23/24.
//

#ifndef TORC_RIGID_BODY_H
#define TORC_RIGID_BODY_H

#include "pinocchio_model.h"
#include "robot_state_types.h"
#include "contact_state.h"

namespace torc::models {
    using vectorx_t = Eigen::VectorXd;
    using matrixx_t = Eigen::MatrixXd;

    class RigidBody : public PinocchioModel{
    public:

        RigidBody(std::string name, std::filesystem::path urdf);

        [[nodiscard]] RobotStateDerivative GetDynamics(const RobotState& state, const vectorx_t& input) const override;

        [[nodiscard]] RobotStateDerivative GetDynamics(const RobotState& state, const vectorx_t& input,
                                                    const ContactState& contacts ) const;

        void DynamicsDerivative(const RobotState& state, const vectorx_t& input,
                                matrixx_t& A, matrixx_t& B) const override;


        void DynamicsDerivative(const RobotState& state, const vectorx_t& input, const ContactState& contacts,
                                matrixx_t& A, matrixx_t& B) const;

    protected:

        matrixx_t ConstraintJacobian(const ContactState& contacts) const;

        vectorx_t ConstraintDrift(const ContactState& contacts) const;

    private:

    };
} // torc::models


#endif //TORC_RIGID_BODY_H
