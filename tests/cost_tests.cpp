#define CATCH_CONFIG_MAIN

#include <catch2/catch_test_macros.hpp>

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <random>
#include "linear_cost.h"
#include "quadratic_cost.h"
#include "explicit_diff_cost.h"

TEST_CASE("Linear Cost Test", "[cost]") {
    int n_tests = 20;
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(1,6);

    SECTION("Provided coefficients") {
        for (int i=0; i<n_tests; i++) {
            size_t dim = dist(rng);
            Eigen::VectorX<double> q = Eigen::VectorX<double>::Random(dim);
            torc::cost::LinearCost<double> cost = torc::cost::LinearCost(q);
            for (int j=0; j<n_tests; j++) {
                Eigen::VectorX<double> v = Eigen::VectorX<double>::Random(dim);
                REQUIRE(cost.Evaluate(v) == q.dot(v));
                REQUIRE(cost.GetCoefficients() == q);
                REQUIRE(cost.Gradient() == q);
                REQUIRE(cost.Hessian(v) == Eigen::MatrixX<double>::Zero(dim, dim));
                REQUIRE(cost.GetDomainDim() == dim);
            }
        }
    }

    SECTION("Default coefficients") {
        for (int dim=0; dim < n_tests; dim++) {
            torc::cost::LinearCost<double> cost = torc::cost::LinearCost<double>(dim);
            Eigen::VectorX<double> zero = Eigen::VectorX<double>::Zero(dim);
            for (int j=0; j<n_tests; j++) {
                Eigen::VectorX<double> v = Eigen::VectorX<double>::Random(dim);
                REQUIRE(cost.Evaluate(v) == 0);
                REQUIRE(cost.GetCoefficients() == zero);
                REQUIRE(cost.Gradient() == zero);
                REQUIRE(cost.Hessian(v) == Eigen::MatrixX<double>::Zero(dim, dim));
                REQUIRE(cost.GetDomainDim() == dim);
            }
        }
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

    torc::cost::ExplicitDifferentialCost<double> cost(func, grad, hess, 3);
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