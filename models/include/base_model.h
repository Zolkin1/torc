#ifndef TORC_BASE_MODEL_H
#define TORC_BASE_MODEL_H

#include <eigen3/Eigen/Dense>
#include <utility>
#include "robot_state_types.h"

namespace torc::models {
    enum SystemType {
        HybridSystemNoImpulse,
        HybridSystemImpulse,
        ContinuousSystem
    };

    class BaseModel {
    public:
        using vectorx_t = Eigen::VectorXd;
        using matrixx_t = Eigen::MatrixXd;

        BaseModel(std::string name);

        std::string GetName() const;

        virtual RobotStateDerivative GetDynamics(const RobotState& state, const vectorx_t& input) const = 0;

        virtual void DynamicsDerivative(const RobotState& state, const vectorx_t& input,
                                        matrixx_t& A, matrixx_t& b) const = 0;

        SystemType GetSystemType() const;
    protected:
        std::string name_;

        SystemType system_type_;

    private:
    };
} // torc::models

#endif //TORC_BASE_MODEL_H