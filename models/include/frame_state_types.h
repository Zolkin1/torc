//
// Created by zolkin on 6/18/24.
//

#ifndef TORC_FRAME_STATE_TYPES_H
#define TORC_FRAME_STATE_TYPES_H

namespace torc::models {
    struct FrameState {
        pinocchio::SE3 placement;
        pinocchio::Motion vel;

        FrameState(const pinocchio::SE3& p, const pinocchio::Motion& v) {
            placement = p;
            vel = v;
        }

        bool operator==(const FrameState& other) const {
            return other.placement == placement && other.vel == vel;
        }
    };
} // torc::models

#endif //TORC_FRAME_STATE_TYPES_H
