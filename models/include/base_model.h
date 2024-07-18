#ifndef TORC_BASE_MODEL_H
#define TORC_BASE_MODEL_H

#include <eigen3/Eigen/Dense>
#include <initializer_list>

namespace torc::models {

    // typedefs
    using vectorx_t = Eigen::VectorXd;
    using matrixx_t = Eigen::MatrixXd;
    using vector3_t = Eigen::Vector3d;
    using quat_t = Eigen::Quaterniond;

    enum SystemType {
        HybridSystemNoImpulse,
        HybridSystemImpulse,
        ContinuousSystem
    };

    class BaseModel {

    public:
      explicit BaseModel(const std::string& name,
                         const SystemType &system_type);

      BaseModel(const BaseModel &other);

      virtual vectorx_t GetDynamics(const vectorx_t &state,
                                    const vectorx_t &input) = 0;

      virtual void DynamicsDerivative(const vectorx_t &state,
                                      const vectorx_t &input, matrixx_t &A,
                                      matrixx_t &b) = 0;

      [[nodiscard]] SystemType GetSystemType() const;

      [[nodiscard]] std::string GetName() const;

    protected:
      // static void HandleParseState(const vectorx_t &state,
                                   // std::initializer_list<vectorx_t> args);

      std::string name_;
      SystemType system_type_;
    };

} // torc::models

#endif //TORC_BASE_MODEL_H