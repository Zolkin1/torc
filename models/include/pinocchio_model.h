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
    using matrix6x_t = Eigen::Matrix<double, 6, Eigen::Dynamic>;

    class PinocchioModel : public BaseModel {
    public:

        static constexpr int FLOATING_CONFIG = 7;
        static constexpr int FLOATING_VEL = 6;

        /**
         * Create the pinocchio model. Assumes every joint that is not fixed and not the root is actuated.
         * @param name Name of the model
         * @param urdf path to the urdf
         */
        PinocchioModel(const std::string& name,
                       const std::filesystem::path& model_path,
                       const SystemType& system_type,
                       bool urdf_model = true);

        PinocchioModel(const PinocchioModel& other);

        [[nodiscard]] virtual vectorx_t InputsToTau(const vectorx_t& input) const = 0;

        [[nodiscard]] long GetNumInputs() const;

        [[nodiscard]] int GetConfigDim() const;

        [[nodiscard]] int GetVelDim() const;

        [[nodiscard]] virtual int GetStateDim() const = 0;

        [[nodiscard]] virtual int GetDerivativeDim() const = 0;

        [[nodiscard]] double GetMass() const;

        [[nodiscard]] int GetNumFrames() const;

        [[nodiscard]] int GetNumJoints() const;

        [[nodiscard]] std::string GetFrameName(int j) const;

        [[nodiscard]] std::string GetFrameType(int j) const;

        [[nodiscard]] long GetFrameIdx(const std::string& frame) const;

        [[nodiscard]] long GetParentJointIdx(const std::string& frame) const;

        [[nodiscard]] vectorx_t GetNeutralConfig() const;

        [[nodiscard]] vectorx_t GetRandomConfig() const;

        [[nodiscard]] vectorx_t GetRandomVel() const;

        [[nodiscard]] virtual vectorx_t GetRandomState() const = 0;

        [[nodiscard]] virtual quat_t GetBaseOrientation(const vectorx_t &q) const = 0;

        // -------------------------------------- //
        // ------------- Kinematics ------------- //
        // -------------------------------------- //
        void FirstOrderFK(const vectorx_t& q);

        void SecondOrderFK(const vectorx_t& q, const vectorx_t& v);

        void ThirdOrderFK(const vectorx_t& q, const vectorx_t& v, const vectorx_t& a);

        // TODO: Should these functions accept frame ID instead?
        /**
         * Calculate the frame state.
         * @param frame name
         * @return frame placement (in world frame) and velocity (in local frame).
         */
        [[nodiscard]] FrameState GetFrameState(const std::string& frame) const;

        /**
         * Calculate the frame state after calling the forward kinematics.
         * @param frame name
         * @param state of the robot
         * @return frame placement (in world frame) and velocity (in local frame).
         */
        FrameState GetFrameState(const std::string& frame, const vectorx_t& q, const vectorx_t& v);

        void GetFrameJacobian(const std::string& frame, const vectorx_t& q, matrix6x_t& J) const;

    protected:

        void MakePinocchioContacts(const RobotContactInfo& contact_info,
                                   std::vector<pinocchio::RigidConstraintModel>& contact_models,
                                   std::vector<pinocchio::RigidConstraintData>& contact_datas) const;

        std::filesystem::path model_path_;

        pinocchio::Model pin_model_;
        std::unique_ptr<pinocchio::Data> pin_data_;

        double mass_;

        long n_input_;

        static const std::string ROOT_JOINT;

    private:
        void CreatePinModel(bool urdf_model);
    };
} // torc::models


#endif //TORC_PINOCCHIOMODEL_H
