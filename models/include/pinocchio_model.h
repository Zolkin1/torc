#ifndef TORC_PINOCCHIOMODEL_H
#define TORC_PINOCCHIOMODEL_H

#include <eigen3/Eigen/Dense>
#include <filesystem>

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/multibody/data.hpp"

#include "base_model.h"
#include "robot_contact_info.h"

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
        PinocchioModel(std::string name, std::filesystem::path urdf,
                       const RobotContactInfo& contact_info);

        /**
         * Create the pinocchio model. User provides a list of joints that are not actuated.
         * @param name
         * @param urdf
         * @param underactuated_joints
         */
        PinocchioModel(std::string name, std::filesystem::path urdf,
                       const std::vector<std::string>& underactuated_joints,
                       const RobotContactInfo& contact_info);

        /**
         * Takes the torques on the actuated coordinates and maps to a vector of
         * dimension model.nv with zeros on underacutated joints
         * @param input
         * @return full input vector
         */
        [[nodiscard]] vectorx_t InputsToFullTau(const vectorx_t& input) const;

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

    protected:

        void CreateActuationMatrix(const std::vector<std::string>& underactuated_joints);

        std::filesystem::path urdf_;

        pinocchio::Model pin_model_;
        std::unique_ptr<pinocchio::Data> pin_data_;

        matrixx_t act_mat_;

        double mass_;

        // TODO: Note that the RigidConstraint* classes are likely to change to just be generic constraint classes
        //  when the pinocchio 3 api is more stabilized. For now this is what we have.
        std::vector<pinocchio::RigidConstraintModel> contact_model_;
        std::vector<pinocchio::RigidConstraintData> contact_data_;

    private:
        void CreatePinModel(const RobotContactInfo& contact_info);
    };
} // torc::models


#endif //TORC_PINOCCHIOMODEL_H
