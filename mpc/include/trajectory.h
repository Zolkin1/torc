//
// Created by zolkin on 6/18/24.
//

#ifndef TORC_TRAJECTORY_H
#define TORC_TRAJECTORY_H

#include "robot_state_types.h"
#include "robot_contact_info.h"

namespace torc::mpc {
    using vectorx_t = Eigen::VectorXd;
    struct Trajectory {
        std::vector<models::RobotState> states;
        std::vector<models::vectorx_t> inputs;
    };

    struct ContactTrajectory : public Trajectory {
        std::vector<models::RobotContactInfo> contacts;
    };
}

#endif //TORC_TRAJECTORY_H
