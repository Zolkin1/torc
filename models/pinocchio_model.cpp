
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/parsers/mjcf.hpp"

#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/model.hpp"

#include "pinocchio_model.h"

#include <utility>

#include "full_order_rigid_body.h"

namespace torc::models {
    const std::string PinocchioModel::ROOT_JOINT = "root_joint";

    PinocchioModel::PinocchioModel(const std::string& name,
                                   const std::filesystem::path& model_path, const SystemType& system_type, bool urdf_model)
        : BaseModel(name, system_type), model_path_(model_path), n_input_(-1) {

        // Create the pinocchio model
        CreatePinModel(urdf_model);

        mass_ = pinocchio::computeTotalMass(pin_model_);
    }

    PinocchioModel::PinocchioModel(const torc::models::PinocchioModel& other)
        : BaseModel(other.name_, other.system_type_) {
        model_path_ = other.model_path_;
        pin_model_ = other.pin_model_;
        pin_data_ = std::make_unique<pinocchio::Data>(*other.pin_data_); // TODO: Check that this works as expected
        mass_ = other.mass_;
        n_input_ = other.n_input_;
    }

    void PinocchioModel::CreatePinModel(bool urdf_model) {
        // Verify that the given file exists
        if (!std::filesystem::exists(model_path_)) {
            throw std::runtime_error("Provided model file does not exist.");
        }

        // Create the pinocchio model from the file
        pin_model_ = pinocchio::Model();

        if (urdf_model) {
            // Verify that we are given a .urdf
            if (model_path_.extension() != ".urdf") {
                throw std::runtime_error("Provided urdf does not end in a .urdf");
            }
            // Normal model
            pinocchio::urdf::buildModel(model_path_, pinocchio::JointModelFreeFlyer(), pin_model_);

            // AD Model
            pin_ad_model_ = pin_model_.cast<torc::ad::adcg_t>();
        } else {
            // Verify that we are given a .xml
            if (model_path_.extension() != ".xml") {
                throw std::runtime_error("Provided urdf does not end in a .xml");
            }
            throw std::runtime_error("MJCF files not fully supported yet.");
            pinocchio::mjcf::buildModel(model_path_, pinocchio::JointModelFreeFlyer(), pin_model_);
        }

        // Normal data
        pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);

        // AD data
        pin_ad_data_ = std::make_shared<ad_pin_data_t>(pin_ad_model_);
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
        return pinocchio::neutral(pin_model_).cwiseMax(GetLowerConfigLimits()).cwiseMin(GetUpperConfigLimits());
    }

    vectorx_t PinocchioModel::GetRandomConfig() const {
        vectorx_t q(GetConfigDim());
        vectorx_t lb = GetLowerConfigLimits();
        vectorx_t ub = GetUpperConfigLimits();
        lb.head<FLOATING_CONFIG>() = vectorx_t::Constant(FLOATING_CONFIG, -10);
        ub.head<FLOATING_CONFIG>() = vectorx_t::Constant(FLOATING_CONFIG, 10);

        pinocchio::randomConfiguration(pin_model_, lb, ub, q);
        return q;
    }

    vectorx_t PinocchioModel::GetRandomVel() const {
        return vectorx_t::Random(GetVelDim());
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

    long PinocchioModel::GetFrameIdx(const std::string& frame) const {
        unsigned long idx = pin_model_.getFrameId(frame);
        if (idx == pin_model_.frames.size()) {
            return -1;
        } else {
            return idx;
        }
    }

    long PinocchioModel::GetParentJointIdx(const std::string& frame) const {
        unsigned long idx = this->pin_model_.frames.at(GetFrameIdx(frame)).parentJoint;
        if (idx == pin_model_.joints.size()) {
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

     void PinocchioModel::FirstOrderFK(const vectorx_t& q) {
         assert(q.size() == this->GetConfigDim());
         pinocchio::framesForwardKinematics(pin_model_, *pin_data_, q);
     }

    void PinocchioModel::SecondOrderFK(const vectorx_t& q, const vectorx_t& v) {
        assert(q.size() == this->GetConfigDim());
        assert(v.size() == this->GetVelDim());
        pinocchio::forwardKinematics(pin_model_, *pin_data_, q, v);
    }

     void PinocchioModel::ThirdOrderFK(const vectorx_t& q, const vectorx_t& v, const vectorx_t& a) {
         assert(q.size() == this->GetConfigDim());
         assert(v.size() == this->GetVelDim());
         assert(a.size() == this->GetVelDim());
         pinocchio::forwardKinematics(pin_model_, *pin_data_, q, v, a);
    }

    FrameState PinocchioModel::GetFrameState(const std::string& frame, const pinocchio::ReferenceFrame& ref) const {
        pinocchio::updateFramePlacements(pin_model_, *pin_data_);
        const long idx = GetFrameIdx(frame);
        if (idx != -1) {
            FrameState state(pin_data_->oMf.at(idx),
                             pinocchio::getFrameVelocity(pin_model_, *pin_data_, idx, ref));
            return state;
        }
        throw std::runtime_error("Provided frame does not exist.");
    }

     FrameState PinocchioModel::GetFrameState(const std::string& frame, const vectorx_t& q, const vectorx_t& v, const pinocchio::ReferenceFrame& ref) {
         SecondOrderFK(q, v);
         return GetFrameState(frame, ref);
     }

    void PinocchioModel::GetFrameJacobian(const std::string& frame, const vectorx_t& q, matrix6x_t& J,
        const pinocchio::ReferenceFrame& ref) const {
        const long idx = GetFrameIdx(frame);
        if (idx == -1) {
            throw std::runtime_error("Provided frame does not exist.");
        }

        J.resize(6, GetVelDim());
        J.setZero();

        pinocchio::computeFrameJacobian(pin_model_, *pin_data_, q, idx, ref, J);
    }

    std::string PinocchioModel::GetUrdfRobotName() const {
        return pin_model_.name;
    }

    vectorx_t PinocchioModel::GetUpperConfigLimits() const {
        vectorx_t q_ub = pin_model_.upperPositionLimit;
        q_ub.head<FLOATING_CONFIG>() = q_ub.head<FLOATING_CONFIG>().cwiseMin(100);
        return q_ub;
    }

    vectorx_t PinocchioModel::GetLowerConfigLimits() const {
        vectorx_t q_lb = pin_model_.lowerPositionLimit;
        q_lb.head<FLOATING_CONFIG>() = q_lb.head<FLOATING_CONFIG>().cwiseMax(-100);
        return q_lb;
    }

    vectorx_t PinocchioModel::GetVelocityJointLimits() const {
        return pin_model_.velocityLimit.cwiseMin(1000); // Bound at 1000 to prevent having crazy large numbers
    }

    vectorx_t PinocchioModel::GetTorqueJointLimits() const {
        return pin_model_.effortLimit.tail(GetNumInputs());
    }

    // ------------------------------------ //
    // ---------- Getters for AD ---------- //
    // ------------------------------------ //
    const ad_pin_model_t& PinocchioModel::GetADPinModel() const {
        return pin_ad_model_;
    }

    std::shared_ptr<ad_pin_data_t> PinocchioModel::GetADPinData() {
        return pin_ad_data_;
    }


} // torc::models
