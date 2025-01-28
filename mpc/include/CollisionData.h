//
// Created by zolkin on 1/28/25.
//

#ifndef COLLISIONDATA_H
#define COLLISIONDATA_H

#include <string>

namespace torc::mpc {
    struct CollisionData {
        std::string frame1;
        std::string frame2;

        double r1;
        double r2;
    };
}    // namespace torc::mpc

#endif //COLLISIONDATA_H
