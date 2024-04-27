#define CATCH_CONFIG_MAIN

#include <catch2/catch_test_macros.hpp>

#include <eigen3/Eigen/Dense>
#include "linear_cost.h"
#include "explicit_diff_cost.h"

TEST_CASE("Linear Cost Test", "[cost]") {
    Eigen::Vector3d q1, x1, x2;
    q1 << 1, 2, 3;
    x1 << 0.1, 1, 10;
    x2 << 0, 0, 0;
    std::string name1 = "linear cost 1";
    torc::LinearCost cost1 = torc::LinearCost<double>(q1, name1);
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
    torc::LinearCost cost2 = torc::LinearCost<float>(q2, name2);
    REQUIRE(cost2.Evaluate(x3) == 26);
    REQUIRE(cost2.Gradient() == q2);
}


TEST_CASE("Explicit Differential Cost Tests") {
    using vectorx_t = Eigen::VectorX<double>;
    using matrixx_t = Eigen::MatrixX<double>;

    std::function<double(vectorx_t)> func = [](const vectorx_t& r) {
        return r[0] * r[1] + r[1] * r[2] + r[2] * r[0];
    };

    std::function<vectorx_t(vectorx_t)> grad = [](const vectorx_t& r) {
        Eigen::Vector3d gradient;
        gradient[0] = r[1] + r[2];
        gradient[1] = r[0] + r[2];
        gradient[2] = r[0] + r[1];
        return gradient;
    };

    std::function<matrixx_t(vectorx_t)> hess = [](const vectorx_t& xyz) {
        Eigen::MatrixXd hessian(3, 3);
        hessian << 0, 1, 1,
                   1, 0, 1,
                   1, 1, 0;
        return hessian;
    };

    torc::ExplicitDifferentialCost<double> cost(func, grad, hess, 3);
    Eigen::Vector3d input = {1, 2, 3};

    SECTION("Evaluation Test") {
        REQUIRE(cost.Evaluate(input) == 11);
    }

    SECTION("Gradient Test") {
        Eigen::Vector3d expected_gradient = {5, 4, 3};
        REQUIRE(cost.Gradient(input).isApprox(expected_gradient));
    }

    SECTION("Hessian Test") {
        Eigen::MatrixXd expected_hessian(3, 3);
        expected_hessian << 0, 1, 1,
                            1, 0, 1,
                            1, 1, 0;
        REQUIRE(cost.Hessian(input).isApprox(expected_hessian));
    }
}