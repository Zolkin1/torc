#define CATCH_CONFIG_MAIN

#include <catch2/catch_test_macros.hpp>

#include <eigen3/Eigen/Dense>
#include "linear_cost.h"
#include "autodiff_cost.h"

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


TEST_CASE("Auto-differentiation Cost Test", "[cost]") {
    namespace ADCG = CppAD::cg;
    namespace AD = CppAD;
    using cg_t = ADCG::CG<double>;  // CodeGen scalar
    using adcg_t = CppAD::AD<cg_t>;       // CppAD scalar templated by CodeGen scalar

    std::function<adcg_t(Eigen::VectorX<adcg_t>)> func = [](const Eigen::VectorX<adcg_t>& r) {
        return r[0] * r[1] + r[1] * r[2] + r[2] * r[0];
    };

    torc::AutodiffCost<double> cost(func, 3);
    SECTION("Valid Names") {
        std::vector<std::string> invalid_names = {"-a", "-", "_-", "-_-", "?", "one2#"};
        for (const std::string& str : invalid_names) {
            REQUIRE(cost.IsValidIdentifier(str) == false);
        }

        std::vector<std::string> valid_names = {"_", "_one", "_one2", "_2", "one"};
        for (const std::string& str : valid_names) {
            REQUIRE(cost.IsValidIdentifier(str) == true);
        }
    }

    Eigen::Vector3d input = {1, 2, 3};

    SECTION("Evaluation") {
        REQUIRE(cost.Evaluate(input) == 11);
    }

    SECTION("Gradient") {
        Eigen::Vector3d expected_gradient = {5, 4, 3};
        REQUIRE(cost.Gradient(input).isApprox(expected_gradient));
    }

    SECTION("Hessian") {
        Eigen::MatrixXd expected_hessian(3, 3);
        expected_hessian << 0, 1, 1,
                            1, 0, 1,
                            1, 1, 0;
        REQUIRE(cost.Hessian(input).isApprox(expected_hessian));
    }
}