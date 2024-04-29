#define CATCH_CONFIG_MAIN

#include <catch2/catch_test_macros.hpp>

#include <eigen3/Eigen/Dense>
#include <iostream>
#include "linear_cost.h"
#include "quadratic_cost.h"

TEST_CASE("Linear Cost Test", "[cost]") {
    Eigen::Vector3d q1, x1, x2;
    q1 << 1, 2, 3;
    x1 << 0.1, 1, 10;
    x2 << 0, 0, 0;
    std::string name1 = "linear cost 1";
    torc::cost::LinearCost cost1 = torc::cost::LinearCost<double>(q1, name1);
    REQUIRE(cost1.Evaluate(x1) == 32.1);
    REQUIRE(cost1.GetDomainDim() == 3);
    REQUIRE(cost1.GetCoefficients() == q1);
    REQUIRE(cost1.Gradient() == q1);
    REQUIRE(cost1.Gradient(x2) == q1);
    REQUIRE(cost1.Hessian(x1) == Eigen::MatrixXd::Zero(3, 3));
    REQUIRE(cost1.GetIdentifier() == name1);

    Eigen::Vector2f q2, x3;
    q2 << 1.5, 2.5;
    x3 << 4, 8;
    std::string name2 = "linear cost 2";
    torc::cost::LinearCost cost2 = torc::cost::LinearCost<float>(q2, name2);
    REQUIRE(cost2.Evaluate(x3) == 26);
    REQUIRE(cost2.Gradient() == q2);
};

TEST_CASE("Quadratic Cost Test", "[cost]") {
    Eigen::Matrix2d A, Au;
    A << 1, 2,
         2, 1;
    Au << 1, 2,
         0, 1;
    Eigen::Matrix4d B, Bu, Bfull;
    B << 1, 0, 0, -3,
         3, 0, 0, 0,
         0.1, 0, 0, 0,
         -1, 0, -10, 0;
    Bu << 1, 3, 0.1, -4,
          0, 0, 0, 0,
          0, 0, 0, -10,
          0, 0, 0, 0;
    Bfull << 1, 3, 0.1, -4,
             3, 0 ,0, 0,
             0.1, 0, 0, -10,
             -4, 0, -10, 0;
    Eigen::Vector2d v1, zero1;
    v1 << 1, 2;
    zero1 << 0, 0;
    std::string name = "quad cost 1";
    std::string name2 = "quad cost 2";
    torc::cost::QuadraticCost cost1 = torc::cost::QuadraticCost<double>(A.triangularView<Eigen::Upper>(), name);
    torc::cost::QuadraticCost cost1_u = torc::cost::QuadraticCost<double>(Au.triangularView<Eigen::Upper>(), name);
    std::cout << cost1.GetQuadCoefficients();
    REQUIRE(Eigen::Matrix2d(cost1.GetQuadCoefficients()) == A);
    REQUIRE(Eigen::Matrix2d(cost1_u.GetQuadCoefficients()) == A);
    REQUIRE(cost1.Evaluate(v1) == 6.5);
    REQUIRE(cost1.Evaluate(zero1) == 0);

    Eigen::Vector2d v1_grad(5, 4);
    REQUIRE(cost1.Gradient(v1) == v1_grad);

    Eigen::Matrix2d cost1_hess;
    cost1_hess << 1, 2,
                  2, 1;
    REQUIRE(cost1.Hessian(v1) == cost1_hess);
    REQUIRE(cost1.Hessian() == cost1_hess);

    torc::cost::QuadraticCost cost2 = torc::cost::QuadraticCost<double>(Bu.triangularView<Eigen::Upper>(), name2);
//    REQUIRE(Eigen::Matrix4d(cost2.GetQuadCoefficients()) == Bfull);
//    REQUIRE(cost2.GetIdentifier() == name2);
//    REQUIRE(cost2.GetDomainDim() == 4);
//
//    Eigen::Vector4d v3 (1, 2, 3, 4);
//    Eigen::Vector4d zero2(0, 0, 0, 0);
//    REQUIRE(cost2.Evaluate(v3) == -129.2);
//    REQUIRE(cost2.Evaluate(zero2) == 0);
}