//
// Created by zolkin on 1/19/25.
//

#ifndef EXTERNALFORCE_H
#define EXTERNALFORCE_H


#include <Eigen/Core>

namespace torc::models {
    template <typename ScalarT>
    class ExternalForce {
    public:
        std::string frame_name;
        Eigen::Vector3<ScalarT> force_linear;

        ExternalForce(const std::string& frame, const Eigen::Vector3<ScalarT>& force) {
            frame_name = frame;
            force_linear = force;
        }
    };
}


#endif //EXTERNALFORCE_H
