//
// Created by zolkin on 5/23/24.
//

#ifndef TORC_ROBOT_STATE_TYPES_H
#define TORC_ROBOT_STATE_TYPES_H

#include <eigen3/Eigen/Dense>
#include <iostream>

namespace torc::models {
    using vectorx_t = Eigen::VectorXd;

    struct RobotStateDerivative {
        vectorx_t v;
        vectorx_t a;

        RobotStateDerivative(int dim) {
            v = vectorx_t::Zero(dim);
            a = vectorx_t::Zero(dim);
        }

        RobotStateDerivative(const vectorx_t& v, const vectorx_t& a) {
            this->v = v;
            this->a = a;
        }

        RobotStateDerivative& operator=(const RobotStateDerivative& other) {
            if (this == &other) {
                return *this;
            }
            v = other.v;
            a = other.a;

            return *this;
        }

        bool operator==(const RobotStateDerivative& other) const {
            return (other.v == v) && (other.a == a);
        }
    };

    // Does not work (never finishes linking)
//    std::ostream& operator<<(std::ostream& os, const RobotStateDerivative& deriv) {
//        os << "v: \n" << deriv.v << "a: \n" << deriv.a << std::endl;
//        return os;
//    }

    struct RobotState {
        vectorx_t q;
        vectorx_t v;

        RobotState(int q_dim, int v_dim) {
            q = vectorx_t::Zero(q_dim);
            v = vectorx_t::Zero(v_dim);
        }

        RobotState(const vectorx_t& q, const vectorx_t& v) {
            this->q = q;
            this->v = v;
        }

        RobotState& operator=(const RobotState& other) {
            if (this == &other) {
                return *this;
            }
            v = other.v;
            q = other.q;

            return *this;
        }

        bool operator==(const RobotState& other) const {
            return other.q == q && other.v == v;
        }
    };

    // Does not work (linker never finishes)
//    std::ostream& operator<<(std::ostream& os, const RobotState& state) {
//        os << "q: \n" << state.q << "v: \n" << state.v << std::endl;
//        return os;
//    }
} // torc::models


#endif //TORC_ROBOT_STATE_TYPES_H
