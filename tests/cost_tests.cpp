#define CATCH_CONFIG_MAIN

#include <catch2/catch_test_macros.hpp>

#include <eigen3/Eigen/Dense>
#include "linear_cost.h"
#include "finite_diff_cost.h"

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


TEST_CASE("Finite Difference Cost Test") {
    SECTION("Basic grad and hess test") {
        std::function<double(Eigen::VectorXd)> fn = [](const Eigen::VectorXd& x) {
            return x.squaredNorm();
        };;
        torc::FiniteDiffCost cost1 = torc::FiniteDiffCost<double>(fn, 2);

        Eigen::Vector2d in = {1, 2};
        REQUIRE(cost1.Evaluate(in) == 5);

        Eigen::Vector2d grad_expected = {2, 4};
        REQUIRE(cost1.Gradient(in).isApprox(grad_expected, 1e-8));

        Eigen::Matrix2d hess_expected;
        hess_expected << 2, 0,
                         0, 2;
        REQUIRE(cost1.Hessian(in).isApprox(hess_expected, 1e-4));
    }

    SECTION("Test different input size") {
        std::function<double(Eigen::VectorXd)> fn = [](const Eigen::VectorXd& x) {
            return x.array().cube().sum();
        };;
        torc::FiniteDiffCost cost2 = torc::FiniteDiffCost<double>(fn, 3);

        Eigen::Vector3d in = {1, 2, 3};
        REQUIRE(cost2.Evaluate(in) == 36);

        Eigen::Vector3d grad_expected = {3, 12, 27};
        REQUIRE(cost2.Gradient(in).isApprox(grad_expected, 1e-7));

        Eigen::Matrix3d hess_expected;
        hess_expected << 6, 0, 0,
                         0, 12, 0,
                         0, 0, 18;
        REQUIRE(cost2.Hessian(in).isApprox(hess_expected, 1e-3));
    }

    SECTION("Test for different input values") {
        std::function<double(Eigen::VectorXd)> fn = [](const Eigen::VectorXd& x) {
            return x.array().abs().sum();
        };;
        torc::FiniteDiffCost cost3 = torc::FiniteDiffCost<double>(fn, 2);

        Eigen::Vector2d in = {-1, 0};
        REQUIRE(cost3.Evaluate(in) == 1);

        Eigen::Vector2d grad_expected = {-1, 0};
        REQUIRE(cost3.Gradient(in).isApprox(grad_expected, 1e-7));
    }

    SECTION ("Test for Hessian with non-zero elements in locations besides the main diagonal") {
        std::function<double(Eigen::VectorXd)> fn = [](Eigen::VectorXd vec) {
            return vec[0] * vec[1] + vec[1] * vec[2] + vec[2] * vec[0];
        };
        torc::FiniteDiffCost cost = torc::FiniteDiffCost<double>(fn, 3);

        Eigen::Vector3d in = {1, 2, 3};
        REQUIRE(cost.Evaluate(in) == 11);

        Eigen::Vector3d grad_expected = {5, 4, 3};
        REQUIRE(cost.Gradient(in).isApprox(grad_expected, 1e-7));

        Eigen::Matrix3d hess_expected;
        hess_expected << 0, 1, 1,
                         1, 0, 1,
                         1, 1, 0;
        REQUIRE(cost.Hessian(in).isApprox(hess_expected, 1e-4));
    }
}