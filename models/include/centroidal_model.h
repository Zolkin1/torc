//
// Created by gavin on 6/24/24.
//

#ifndef CENTROIDAL_MODEL_H
#define CENTROIDAL_MODEL_H

#include "pinocchio_model.h"

namespace torc::model {
    class CentroidalModel: public models::PinocchioModel {
        public:
            using vectorx_t = Eigen::VectorXd;
            using matrixx_t = Eigen::MatrixXd;
    };
}

#endif //CENTROIDAL_MODEL_H
