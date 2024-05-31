
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"

#include "pinocchio_model.h"

namespace torc::models {
    PinocchioModel::PinocchioModel(std::string name, std::filesystem::path urdf,
                                   const RobotContactInfo& contact_info)
        : BaseModel(std::move(name)), urdf_(std::move(urdf)) {

        // Create the pinocchio model
        CreatePinModel(contact_info);

        // Assuming all joints (not "root_joint") are actuated
        std::vector<std::string> underactuated_joints;
        underactuated_joints.push_back("root_joint");

        CreateActuationMatrix(underactuated_joints);
    }

    PinocchioModel::PinocchioModel(std::string name, std::filesystem::path urdf,
                                   const std::vector<std::string>& underactuated_joints,
                                   const RobotContactInfo& contact_info)
        : BaseModel(std::move(name)), urdf_(std::move(urdf)) {
        // Create the pinocchio model
        CreatePinModel(contact_info);


        CreateActuationMatrix(underactuated_joints);
    }

    void PinocchioModel::CreatePinModel(const RobotContactInfo& contact_info) {
        // Verify that the given file exists
        if (!std::filesystem::exists(urdf_)) {
            throw std::runtime_error("Provided urdf file does not exist.");
        }

        // TODO: Provide support for mujoco format too
        // Verify that we are given a .urdf
        if (urdf_.extension() != ".urdf") {
            throw std::runtime_error("Provided urdf does not end in a .urdf");
        }

        // Create the pinocchio model from the urdf
        pin_model_ = pinocchio::Model();
        pinocchio::urdf::buildModel(urdf_, pinocchio::JointModelFreeFlyer(), pin_model_);

        pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);

        mass_ = pinocchio::computeTotalMass(pin_model_);

        // -------------------------------------------------- //
        // -------------------- Contacts -------------------- //
        // -------------------------------------------------- //
        // Go through each contact and make a pinocchio model
        for (const auto& contact : contact_info.contacts) {
            // Validate the contact frame
            int frame_idx = -1;
            for (int i = 0; i < pin_model_.nframes; i++) {
                if (contact.first == pin_model_.frames.at(i).name) {
                    frame_idx = i;
                }
            }

            if (frame_idx == -1) {
                throw std::runtime_error("Contact frame provided does not match any robot frames.");
            }

            // Use the parentJoint of the frame
            const unsigned long joint_idx = pin_model_.frames.at(frame_idx).parentJoint;

            // Get the contact location in the joint frame
            const pinocchio::SE3 location = pin_model_.frames.at(frame_idx).placement;
            std::cout << "location of " << contact.first << ": " << location << std::endl;

            // Get the contact type
            pinocchio::ContactType ct = pinocchio::CONTACT_UNDEFINED;
            switch (contact.second.type) {
                case PointContact:
                    ct = pinocchio::CONTACT_3D;
                    break;
                case PatchContact:
                    ct = pinocchio::CONTACT_6D;
                    break;
                default:
                    throw std::runtime_error("Invalid contact type provided.");
            }

            contact_model_.emplace_back(ct, pin_model_, joint_idx, location);
        }
    }

    vectorx_t PinocchioModel::InputsToFullTau(const vectorx_t& input) const {
        assert(input.size() == act_mat_.cols());
        return act_mat_*input;
    }

    long PinocchioModel::GetNumInputs() const {
        return act_mat_.cols();
    }

    int PinocchioModel::GetConfigDim() const {
        return pin_model_.nq;
    }

    int PinocchioModel::GetVelDim() const {
        return pin_model_.nv;
    }

    int PinocchioModel::GetStateDim() const {
        return GetConfigDim() + GetVelDim();
    }

    int PinocchioModel::GetDerivativeDim() const {
        return 2*GetVelDim();
    }

    double PinocchioModel::GetMass() const {
        return mass_;
    }

    int PinocchioModel::GetNumFrames() const {
        return pin_model_.nframes;
    }

    int PinocchioModel::GetNumJoints() const {
        return pin_model_.njoints;
    }

    std::string PinocchioModel::GetFrameName(int j) const {
        return pin_model_.frames.at(j).name;
    }

    std::string PinocchioModel::GetFrameType(int j) const {
        switch (pin_model_.frames.at(j).type) {
            case pinocchio::OP_FRAME:
                return "operational_frame";
            case pinocchio::JOINT:
                return "joint";
            case pinocchio::FIXED_JOINT:
                return "fixed_joint";
            case pinocchio::BODY:
                return "body";
            case pinocchio::SENSOR:
                return "sensor";
        }

        throw std::runtime_error("Invalid return type from pinocchio.");
    }

    void PinocchioModel::CreateActuationMatrix(const std::vector<std::string>& underactuated_joints) {
        assert(pin_model_.idx_vs.at(1) == 0);
        assert(pin_model_.nvs.at(1) == FLOATING_VEL);

        int num_actuators = pin_model_.nv;

        std::vector<int> unact_joint_idx;

        unact_joint_idx.push_back(0);   // Universe joint is never actuated

        // Get the number of actuators
        for (std::string joint_name : underactuated_joints) {
            for (int i = 0; i < GetNumJoints(); i++) {
                if (joint_name == pin_model_.names.at(i)) {
                    num_actuators -= pin_model_.joints.at(i).nv();
                    unact_joint_idx.push_back(i);
                    break;
                }
            }
        }

        act_mat_ = matrixx_t::Zero(pin_model_.nv, num_actuators);
        int act_idx = 0;

        for (int joint_idx = 0; joint_idx < GetNumJoints(); joint_idx++) {
            bool act = true;
            for (int idx : unact_joint_idx) {
                if (joint_idx == idx) {
                    act = false;
                    break;
                }
            }

            if (act) {
                const int nv = pin_model_.joints.at(joint_idx).nv();
                act_mat_.block(pin_model_.joints.at(joint_idx).idx_v(), act_idx, nv, nv) =
                        matrixx_t::Identity(nv, nv);
                act_idx += nv;
            }
        }
    }

} // torc::models
