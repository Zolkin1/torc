#ifndef TORC_BASE_MODEL_H
#define TORC_BASE_MODEL_H

#include <eigen3/Eigen/Dense>
#include <utility>
#include "robot_state.h"

namespace torc::models {
    class BaseModel {
    public:
        using vectorx_t = Eigen::VectorXd;
        using matrixx_t = Eigen::MatrixXd;

        BaseModel(std::string name);

        std::string GetName() const;

        virtual vectorx_t GetDynamics(const RobotState& state, const vectorx_t& input) const = 0;

        virtual matrixx_t Dfdx(const RobotState& state, const vectorx_t& input) const = 0;

        virtual matrixx_t Dfdu(const RobotState& state, const vectorx_t& input) const = 0;
    protected:
        std::string name_;
    private:
    };
} // torc::models

#endif //TORC_BASE_MODEL_H