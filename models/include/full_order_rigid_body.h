//
// Created by zolkin on 5/23/24.
//

#ifndef TORC_FULL_ORDER_RIGID_BODY_H
#define TORC_FULL_ORDER_RIGID_BODY_H

#include "pinocchio_model.h"
#include "robot_contact_info.h"

#include <cppad/utility/vector.hpp>

namespace torc::models {
    using vectorx_t = Eigen::VectorXd;
    using matrixx_t = Eigen::MatrixXd;

    class FullOrderRigidBody : public PinocchioModel {
    public:

        FullOrderRigidBody(const std::string& name, const std::filesystem::path& urdf);

        FullOrderRigidBody(const std::string& name, const std::filesystem::path& urdf,
                           const std::vector<std::string>& underactuated_joints);

        FullOrderRigidBody(const FullOrderRigidBody& other);

        [[nodiscard]] int GetStateDim() const override;

        [[nodiscard]] int GetDerivativeDim() const override;

        [[nodiscard]] vectorx_t GetRandomState() const override;

        // @note These are not actually const functions as we modify the pin_data struct
        [[nodiscard]] vectorx_t GetDynamics(const vectorx_t& state,
                                            const vectorx_t& input) override;

        [[nodiscard]] vectorx_t GetDynamics(const vectorx_t& state,
                                            const vectorx_t& input,
                                            const RobotContactInfo& contact_info) const;

        vectorx_t GetImpulseDynamics(const vectorx_t& state,
                                     const vectorx_t& input,
                                     const RobotContactInfo& contact_info);

        // TODO: Linearization function that calculates the FD and derivatives so we don't have redundant calls

        void DynamicsDerivative(const vectorx_t& state,
                                const vectorx_t& input,
                                matrixx_t& A,
                                matrixx_t& B) override;


        void DynamicsDerivative(const vectorx_t& state,
                                const vectorx_t& input,
                                const RobotContactInfo& contacts,
                                matrixx_t& A,
                                matrixx_t& B);


        void ImpulseDerivative(const vectorx_t& state,
                               const vectorx_t& input,
                               const RobotContactInfo& contact_info,
                               matrixx_t& A,
                               matrixx_t& B);

        void ParseState(const vectorx_t& state,
                        vectorx_t& q,
                        vectorx_t& v) const;

        void ParseStateDerivative(const vectorx_t& dstate,
                                  vectorx_t& v,
                                  vectorx_t& a) const;

        static vectorx_t BuildState(const vectorx_t& q, const vectorx_t& v);

        static vectorx_t BuildStateDerivative(const vectorx_t& v, const vectorx_t& a);

        void ParseInput(const vectorx_t& input, vectorx_t& tau) const;

        /**
         * Takes the torques on the actuated coordinates and maps to a vector of
         * dimension model.nv with zeros on underacutated joints
         * @param input
         * @return full input vector
         */
        [[nodiscard]] vectorx_t InputsToTau(const vectorx_t& input) const override;

        constexpr static size_t STATE_Q_IDX = 0;
        constexpr static size_t STATE_V_IDX = 1;

    protected:
        void CreateActuationMatrix(const std::vector<std::string>& underactuated_joints);

        matrixx_t act_mat_;

        std::unique_ptr<pinocchio::Data> contact_data_;
    private:

    };
} // torc::models


#endif //TORC_FULL_ORDER_RIGID_BODY_H
