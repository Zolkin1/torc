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
        double dt;  // Time difference between each node

        virtual void Reset(int nodes, int q_dim, int v_dim);
    };

    // TODO: consider making this encapsulation rather than inheritence
    struct ContactTrajectory : public Trajectory {
        std::vector<models::RobotContactInfo> contacts;
        std::vector<double> impulse_times;

        void Reset(int nodes, int q_dim, int v_dim) override;
    };
}

#endif //TORC_TRAJECTORY_H
