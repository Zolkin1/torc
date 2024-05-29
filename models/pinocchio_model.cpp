
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"

#include "pinocchio_model.h"

namespace torc::models {
    PinocchioModel::PinocchioModel(std::string name, std::filesystem::path urdf)
        : BaseModel(std::move(name)), urdf_(std::move(urdf)) {
        // Verify that the given file exists
        if (!std::filesystem::exists(urdf_)) {
            throw std::runtime_error("Provided urdf file does not exist.");
        }

        // Verify that we are given a .urdf
        if (urdf_.extension() != ".urdf") {
            throw std::runtime_error("Provided urdf does not end in a .urdf");
        }

        // Create the pinocchio model from the urdf
        pinocchio::urdf::buildModel(urdf_, pinocchio::JointModelFreeFlyer(), pin_model_, false);

        pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);

        mass_ = pinocchio::computeTotalMass(pin_model_);
    }

    vectorx_t PinocchioModel::InputsToFullTau(const vectorx_t& input) const {
        // TODO: Implement correctly
        std::cerr << "Function not fully implemented." << std::endl;

        return vectorx_t::Zero(pin_model_.nv);
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

} // torc::models
