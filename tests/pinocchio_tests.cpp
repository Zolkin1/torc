//
// Created by gavin on 6/23/24.
//

#include <iostream>

#include "eigen3/Eigen/Dense"
#include "pinocchio/multibody/sample-models.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea.hpp"

int main()
{
    pinocchio::Model model;
    pinocchio::buildModels::manipulator(model);
    pinocchio::Data data(model);

    Eigen::VectorXd q = neutral(model);
    Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
    Eigen::VectorXd a = Eigen::VectorXd::Zero(model.nv);

    std::cout << "number of joint positions = " << model.nq << "\n";
    std::cout << "number of joint velocities = " << model.nv << "\n";

    rnea(model, data, q, v, a);
    std::cout << "tau = " << data.tau.transpose() << std::endl;
}