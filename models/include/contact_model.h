//
// Created by zolkin on 6/18/24.
//

#ifndef TORC_CONTACT_MODEL_H
#define TORC_CONTACT_MODEL_H

#include "robot_contact_info.h"
#include "base_model.h"

namespace torc::models {
    class ContactModel {
    public:
        virtual vectorx_t GetDynamics(const vectorx_t& state,
                                      const vectorx_t& input,
                                      const RobotContactInfo& contact_info) = 0;

        virtual void DynamicsDerivative(const vectorx_t& state,
                                        const vectorx_t& input,
                                        const RobotContactInfo& contacts,
                                        matrixx_t& A, matrixx_t& B) = 0;
    };
} // torc::models

#endif //TORC_CONTACT_MODEL_H
