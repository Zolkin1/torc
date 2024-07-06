
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/model.hpp"

#include "pinocchio_model.h"

#include <utility>

namespace torc::models {
    const std::string PinocchioModel::ROOT_JOINT = "root_joint";

    PinocchioModel::PinocchioModel(const std::string& name,
                                   const std::filesystem::path& urdf)
        : BaseModel(name), urdf_(std::move(urdf)), n_input_(-1) {

        // Create the pinocchio model
        CreatePinModel();

        mass_ = pinocchio::computeTotalMass(pin_model_);
    }

    PinocchioModel::PinocchioModel(const torc::models::PinocchioModel& other)
        : BaseModel(other.name_) {
        this->system_type_ = other.system_type_;

        urdf_ = other.urdf_;
        pin_model_ = other.pin_model_;
        pin_data_ = std::make_unique<pinocchio::Data>(*other.pin_data_); // TODO: Check that this works as expected
        mass_ = other.mass_;
        n_input_ = other.n_input_;
    }

    void PinocchioModel::CreatePinModel() {
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
    }

    long PinocchioModel::GetNumInputs() const {
        return n_input_;
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

    vectorx_t PinocchioModel::GetNeutralConfig() const {
        return pinocchio::neutral(pin_model_);
    }

    vectorx_t PinocchioModel::GetRandomConfig() const {
        // Make dummy limits
        const vectorx_t ub = vectorx_t::Constant(GetConfigDim(), 10);
        const vectorx_t lb = vectorx_t::Constant(GetConfigDim(), -10);

        vectorx_t q(GetConfigDim());
        pinocchio::randomConfiguration(pin_model_, lb, ub, q);

        return q;
    }

    vectorx_t PinocchioModel::GetRandomVel() const {
        return vectorx_t::Random(GetVelDim());
    }

    vectorx_t PinocchioModel::GetRandomState() const {
        vectorx_t x(GetConfigDim() + GetVelDim());
        x << GetRandomConfig(), GetRandomVel();
        return x;
    }

    std::string PinocchioModel::GetFrameType(const int j) const {
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

    unsigned long PinocchioModel::GetFrameIdx(const std::string& frame) const {
        unsigned long idx = pin_model_.getFrameId(frame);
        if (idx == pin_model_.frames.size()) {
            return -1;
        } else {
            return idx;
        }
    }

    void PinocchioModel::MakePinocchioContacts(const RobotContactInfo& contact_info,
                                               std::vector<pinocchio::RigidConstraintModel>& contact_models,
                                               std::vector<pinocchio::RigidConstraintData>& contact_datas) const {
        // -------------------------------------------------- //
        // -------------------- Contacts -------------------- //
        // -------------------------------------------------- //

        // Clear the vectors
        contact_models.clear();
        contact_datas.clear();

        // Go through each contact and make a pinocchio model
        for (const auto& contact : contact_info.contacts) {
            // Only look at the contacts that are in contact
            if (contact.second.state) {
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

                contact_models.emplace_back(ct, pin_model_, joint_idx, location);
                contact_datas.emplace_back(*(contact_models.end()-1));
            }
        }
    }

    // void PinocchioModel::ForwardKinematics(const vectorx_t& q) {
    //     pinocchio::framesForwardKinematics(pin_model_, *pin_data_, q);
    // }
    //
    // void PinocchioModel::ForwardKinematics(const vectorx_t& state) {
    //     vectorx_t q, v;
    //     ParseState(state, q, v);
    //     pinocchio::forwardKinematics(pin_model_, *pin_data_, q, v);
    // }
    //
    // void PinocchioModel::ForwardKinematics(const vectorx_t& state, const vectorx_t& deriv) {
    //     pinocchio::forwardKinematics(pin_model_, *pin_data_, state.q, state.v, deriv.a);
    // }

    FrameState PinocchioModel::GetFrameState(const std::string& frame) const {
        const unsigned long idx = GetFrameIdx(frame);
        if (idx != -1) {
            FrameState state(pin_data_->oMf.at(idx),
                             pinocchio::getFrameVelocity(pin_model_, *pin_data_, idx));
            return state;
        } else {
            throw std::runtime_error("Provided frame does not exist.");
        }
    }

    // FrameState PinocchioModel::GetFrameState(const std::string& frame, const vectorx_t& state) {
    //     ForwardKinematics(state);
    //     return GetFrameState(frame);
    // }

    void PinocchioModel::GetFrameJacobian(const std::string& frame, const vectorx_t& q, matrixx_t& J) const {
        const unsigned long idx = GetFrameIdx(frame);
        if (idx == -1) {
            throw std::runtime_error("Provided frame does not exist.");
        }

        J.setZero();

        pinocchio::computeFrameJacobian(pin_model_, *pin_data_, q, idx, J);
    }

} // torc::models
