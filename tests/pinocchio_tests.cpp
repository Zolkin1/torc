//
// Created by gavin on 6/23/24.
//

#include <iostream>

#include "pinocchio/multibody/sample-models.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea.hpp"

int main()
{
    pinocchio::Model model;
    pinocchio::buildModels::manipulator(model);
    pinocchio::Data data(model);

    Eigen::VectorXd q = pinocchio::neutral(model);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model.nv);
    Eigen::VectorXd a = Eigen::VectorXd::Zero(model.nv);

    const Eigen::VectorXd & tau = pinocchio::rnea(model, data, q, v, a);
    std::cout << "tau = " << tau.transpose() << std::endl;
}