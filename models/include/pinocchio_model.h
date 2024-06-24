#ifndef TORC_PINOCCHIOMODEL_H
#define TORC_PINOCCHIOMODEL_H

#include <eigen3/Eigen/Dense>
#include <filesystem>

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/multibody/data.hpp"

#include "base_model.h"
#include "robot_contact_info.h"
#include "frame_state_types.h"

namespace torc::models {

    class PinocchioModel : public BaseModel {
    public:
        using vectorx_t = Eigen::VectorXd;
        using matrixx_t = Eigen::MatrixXd;

        static constexpr int FLOATING_CONFIG = 7;
        static constexpr int FLOATING_VEL = 6;

        /**
         * Create the pinocchio model. Assumes every joint that is not fixed and not the root is actuated.
         * @param name Name of the model
         * @param urdf path to the urdf
         */
        PinocchioModel(const std::string& name, std::filesystem::path  urdf);

        [[nodiscard]] virtual vectorx_t InputsToTau(const vectorx_t& input) const = 0;

        [[nodiscard]] long GetNumInputs() const;

        [[nodiscard]] int GetConfigDim() const;

        [[nodiscard]] int GetVelDim() const;

        [[nodiscard]] int GetStateDim() const;

        [[nodiscard]] int GetDerivativeDim() const;

        [[nodiscard]] double GetMass() const;

        [[nodiscard]] int GetNumFrames() const;

        [[nodiscard]] int GetNumJoints() const;

        [[nodiscard]] std::string GetFrameName(int j) const;

        [[nodiscard]] std::string GetFrameType(int j) const;

        [[nodiscard]] unsigned long GetFrameIdx(const std::string& frame) const;

        void GetNeutralConfig(vectorx_t& q) const;

        [[nodiscard]] vectorx_t GetRandomConfig() const;

        [[nodiscard]] vectorx_t GetRandomVel() const;

        [[nodiscard]] RobotState GetRandomState() const;

        // -------------------------------------- //
        // ------------- Kinematics ------------- //
        // -------------------------------------- //
        void ForwardKinematics(const vectorx_t& q);

        void ForwardKinematics(const RobotState& state);

        void ForwardKinematics(const RobotState& state, const RobotStateDerivative& deriv);

        // TODO: Should these functions accept frame ID instead?
        /**
         * Calculate the frame state.
         * @param frame name
         * @return frame placement (in world frame) and velocity (in local frame).
         */
        FrameState GetFrameState(const std::string& frame);

        /**
         * Calculate the frame state after calling the forward kinematics.
         * @param frame name
         * @param state of the robot
         * @return frame placement (in world frame) and velocity (in local frame).
         */
        FrameState GetFrameState(const std::string& frame, const RobotState& state);

        void GetFrameJacobian(const std::string& frame, const vectorx_t& q, matrixx_t& J);

    protected:

        void MakePinocchioContacts(const RobotContactInfo& contact_info,
                                   std::vector<pinocchio::RigidConstraintModel>& contact_models,
                                   std::vector<pinocchio::RigidConstraintData>& contact_datas) const;

        std::filesystem::path urdf_;

        pinocchio::Model pin_model_;
        std::unique_ptr<pinocchio::Data> pin_data_;

        double mass_;

        long num_inputs_;

        static const std::string ROOT_JOINT;

    private:
        void CreatePinModel();
    };
} // torc::models


#endif //TORC_PINOCCHIOMODEL_H
