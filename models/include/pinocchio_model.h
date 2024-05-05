#ifndef TORC_PINOCCHIOMODEL_H
#define TORC_PINOCCHIOMODEL_H

#include <eigen3/Eigen/Dense>
#include <filesystem>

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/multibody/data.hpp"

#include "base_model.h"

namespace torc::models {

    class PinocchioModel : public BaseModel {
        using vectorx_t = Eigen::VectorXd;

    public:
        PinocchioModel(std::string name, std::filesystem::path urdf);
    protected:
        std::filesystem::path urdf_;

        pinocchio::Model pin_model_;
        std::unique_ptr<pinocchio::Data> pin_data_;

    private:
    };
} // torc::models


#endif //TORC_PINOCCHIOMODEL_H
