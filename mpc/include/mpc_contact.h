//
// Created by zolkin on 6/18/24.
//

#ifndef TORC_MPC_CONTACT_H
#define TORC_MPC_CONTACT_H

#include <memory>

#include "contact_model.h"
#include "rigid_body.h"

#include "trajectory.h"

namespace torc::mpc {
    class MPCContact {
    public:
        MPCContact(const models::ContactModel& model);

        // Want to provide this function for all the generic data types and any of the special ones
        //  that I provide an interface for.
        void ToHPIPMData(const Trajectory& traj);

    protected:
       models::RigidBody model_;

       // cost fcn

       // constraints

    private:
    };
} // torc::mpc


#endif //TORC_MPC_CONTACT_H
