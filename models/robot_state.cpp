//
// Created by zolkin on 5/23/24.
//

#include "robot_state.h"

#include <utility>

namespace torc::models {
    RobotState::RobotState(vectorx_t& q,
                           vectorx_t& v)
        : q_(std::move(q)), v_(std::move(v)) {}

    RobotState& RobotState::operator=(const torc::models::RobotState& state) {
        if (this == &state) {
            return *this;
        }

        q_ = state.q_;
        v_ = state.v_;

        return *this;
    }

    void RobotState::Setq(const vectorx_t& q) {
        q_ = q;
    }

    void RobotState::Setv(const vectorx_t& v) {
        v_ = v;
    }

    vectorx_t RobotState::Getq() const {
        return q_;
    }

    vectorx_t RobotState::Getv() const {
        return v_;
    }

    const vectorx_t& RobotState::GetRefq() const {
        return q_;
    }

    const vectorx_t& RobotState::GetRefv() const {
        return v_;
    }

} // torc::models