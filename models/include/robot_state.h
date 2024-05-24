//
// Created by zolkin on 5/23/24.
//

#ifndef TORC_ROBOT_STATE_H
#define TORC_ROBOT_STATE_H

#include <eigen3/Eigen/Dense>

namespace torc::models {
    using vectorx_t = Eigen::VectorXd;

    class RobotState {

    public:
        RobotState(vectorx_t& q, vectorx_t& v);

        RobotState& operator=(const RobotState& state);

        void Setq(const vectorx_t& q);

        void Setv(const vectorx_t& v);

        vectorx_t Getq() const;

        vectorx_t Getv() const;

        const vectorx_t& GetRefq() const;

        const vectorx_t& GetRefv() const;

    protected:
    private:
        vectorx_t q_;
        vectorx_t v_;
    };
} // torc::models


#endif //TORC_ROBOT_STATE_H
