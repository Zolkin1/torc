#ifndef TORC_PINOCCHIOMODEL_H
#define TORC_PINOCCHIOMODEL_H

#include <eigen3/Eigen/Dense>
#include <filesystem>

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/multibody/data.hpp"

#include "base_model.h"

namespace torc::models {

    class PinocchioModel : public BaseModel {
    public:
        using vectorx_t = Eigen::VectorXd;
        using matrixx_t = Eigen::MatrixXd;

        /**
         * Create the pinocchio model
         * @param name Name of the model
         * @param urdf path to the urdf
         */
        PinocchioModel(std::string name, std::filesystem::path urdf);

    protected:
        std::filesystem::path urdf_;

        pinocchio::Model pin_model_;
        std::unique_ptr<pinocchio::Data> pin_data_;

    private:
    };
} // torc::models


#endif //TORC_PINOCCHIOMODEL_H
