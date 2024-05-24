//
// Created by zolkin on 5/23/24.
//

#ifndef TORC_RIGID_BODY_H
#define TORC_RIGID_BODY_H

#include "pinocchio_model.h"
#include "robot_state.h"
#include "contact_state.h"

namespace torc::models {
    using vectorx_t = Eigen::VectorXd;
    using matrixx_t = Eigen::MatrixXd;

    class RigidBody : public PinocchioModel{
    public:

        RigidBody(std::string name, std::filesystem::path urdf);

        [[nodiscard]] vectorx_t GetDynamics(const RobotState& state, const vectorx_t& input) const override;

        [[nodiscard]] vectorx_t GetDynamicsContacts(const RobotState& state, const vectorx_t& input,
                                                    const ContactState& contacts ) const;

        matrixx_t Dfdx(const RobotState& state, const vectorx_t& input) const override;
        matrixx_t Dfdu(const RobotState& state, const vectorx_t& input) const override;
    protected:

        matrixx_t ConstraintJacobian(const ContactState& contacts) const;
        vectorx_t ConstraintDrift(const ContactState& contacts) const;

    private:

    };
} // torc::models


#endif //TORC_RIGID_BODY_H
