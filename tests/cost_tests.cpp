#define CATCH_CONFIG_MAIN

#include <catch2/catch_test_macros.hpp>

#include <eigen3/Eigen/Dense>
#include <random>
#include "linear_cost.h"
#include "quadratic_cost.h"
#include "analytic_cost.h"
#include "finite_diff_cost.h"

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
}


TEST_CASE("Finite Difference Cost Test") {
    SECTION("Basic grad and hess test") {
        std::function<double(Eigen::VectorXd)> fn = [](const Eigen::VectorXd& x) {
            return x.squaredNorm();
        };;
        torc::cost::FiniteDiffCost cost1 = torc::cost::FiniteDiffCost<double>(fn, 2);

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
        };

        torc::cost::FiniteDiffCost cost2 = torc::cost::FiniteDiffCost<double>(fn, 3);

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
        torc::cost::FiniteDiffCost cost3 = torc::cost::FiniteDiffCost<double>(fn, 2);

        Eigen::Vector2d in = {-1, 0};
        REQUIRE(cost3.Evaluate(in) == 1);

        Eigen::Vector2d grad_expected = {-1, 0};
        REQUIRE(cost3.Gradient(in).isApprox(grad_expected, 1e-7));
    }

    SECTION ("Test for Hessian with non-zero elements in locations besides the main diagonal") {
        std::function<double(Eigen::VectorXd)> fn = [](Eigen::VectorXd vec) {
            return vec[0] * vec[1] + vec[1] * vec[2] + vec[2] * vec[0];
        };
        torc::cost::FiniteDiffCost cost = torc::cost::FiniteDiffCost<double>(fn, 3);

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

TEST_CASE("Quadratic Cost Test", "[cost]") {
    int n_tests = 20;
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(1,6);

    SECTION("Full matrix") {
        for (int i=0; i<n_tests; i++) {
            size_t dim = dist(rng);
            Eigen::MatrixX<double> A = Eigen::MatrixX<double>::Random(dim, dim).selfadjointView<Eigen::Upper>();
            torc::cost::QuadraticCost<double> cost = torc::cost::QuadraticCost(A);
            for (int j = 0; j < n_tests; ++j) {
                Eigen::VectorX<double> v = Eigen::VectorX<double>::Random(dim);
                REQUIRE(cost.Evaluate(v) == 0.5 * (v.dot(A * v)));
                REQUIRE(cost.GetQuadCoefficients() == A);
                REQUIRE(cost.GetLinCoefficients() == Eigen::VectorX<double>::Zero(dim));
                REQUIRE(cost.Gradient(v) == A*v);
                REQUIRE(cost.Hessian(v) == A);
                REQUIRE(cost.GetDomainDim() == dim);
            }
        }
    }

    SECTION("Triangular View") {    // matrix has to be fixed-size at compile time to call triangularView
        for (int i=0; i<n_tests; i++) {
            Eigen::Matrix3d A = Eigen::Matrix3d::Random();
            Eigen::TriangularView<Eigen::Matrix3d, Eigen::Upper> Au = A.triangularView<Eigen::Upper>();
            torc::cost::QuadraticCost<double> cost = torc::cost::QuadraticCost(Au);
            A = A.selfadjointView<Eigen::Upper>();
            for (int j = 0; j < n_tests; ++j) {
                Eigen::Vector3d v = Eigen::Vector3d::Random();
                REQUIRE(abs(cost.Evaluate(v) - 0.5 * (v.dot(A * v))) < 1e-8);
                REQUIRE(cost.GetQuadCoefficients() == A);
                REQUIRE(cost.GetLinCoefficients() == Eigen::Vector3d::Zero());
                REQUIRE(cost.Gradient(v).isApprox(A*v));
                REQUIRE(cost.Hessian(v).isApprox(A));
                REQUIRE(cost.GetDomainDim() == 3);
            }
        }
    }

    SECTION("Full matrix with linear cost") {
        for (int i=0; i<n_tests; i++) {
            size_t dim = dist(rng);
            Eigen::MatrixX<double> A = Eigen::MatrixX<double>::Random(dim, dim).selfadjointView<Eigen::Upper>();
            Eigen::VectorX<double> q = Eigen::VectorX<double>::Random(dim);
            torc::cost::QuadraticCost<double> cost = torc::cost::QuadraticCost(A, q);
            for (int j = 0; j < n_tests; ++j) {
                Eigen::VectorX<double> v = Eigen::VectorX<double>::Random(dim);
                REQUIRE(cost.Evaluate(v) == 0.5 * (v.dot(A * v)) + q.dot(v));
                REQUIRE(cost.GetQuadCoefficients() == A);
                REQUIRE(cost.GetLinCoefficients() == q);
                REQUIRE(cost.Gradient(v) == A*v + q);
                REQUIRE(cost.Hessian(v) == A);
                REQUIRE(cost.Hessian() == A);
                REQUIRE(cost.GetDomainDim() == dim);
            }
        }
    }
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

    torc::cost::AnalyticCost<double> cost(func, grad, hess, 3);
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