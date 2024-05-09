#define CATCH_CONFIG_MAIN

#include <random>
#include <iostream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/benchmark/catch_constructor.hpp>
#include <eigen3/Eigen/Dense>
#include <cppad/cg.hpp>

#include "linear_cost.h"
#include "quadratic_cost.h"
#include "autodiff_cost.h"
#include "analytic_cost.h"
#include "finite_diff_cost.h"
#include "test_fn.h"


TEST_CASE("Linear Cost Test", "[linear]") {
    Eigen::Vector3d q1, x1, x2;
    q1 << 1, 2, 3;
    x1 << 0.1, 1, 10;
    x2 << 0, 0, 0;
    std::string name1 = "LinearCost1";
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
    std::string name2 = "LinearCost2";
    torc::cost::LinearCost cost2 = torc::cost::LinearCost<float>(q2, name2);
    REQUIRE(cost2.Evaluate(x3) == 26);
    REQUIRE(cost2.Gradient() == q2);
}


TEST_CASE("Quadratic Cost Test", "[quadratic]") {
    using namespace torc::cost;
    const int n_tests = 10;
    const std::vector<int> test_dims = {1, 50};

    SECTION("Full matrix") {
        for (int i=0; i<n_tests; i++) {
            for (auto dim : test_dims) {
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
            for (auto dim: test_dims) {
                Eigen::MatrixX<double> A = Eigen::MatrixX<double>::Random(dim, dim).selfadjointView<Eigen::Upper>();
                Eigen::VectorX<double> q = Eigen::VectorX<double>::Random(dim);
                torc::cost::QuadraticCost<double> cost = torc::cost::QuadraticCost(A, q);
                for (int j = 0; j < n_tests; ++j) {
                    Eigen::VectorX<double> v = Eigen::VectorX<double>::Random(dim);
                    REQUIRE(cost.Evaluate(v) == 0.5 * (v.dot(A * v)) + q.dot(v));
                    REQUIRE(cost.GetQuadCoefficients() == A);
                    REQUIRE(cost.GetLinCoefficients() == q);
                    REQUIRE(cost.Gradient(v) == A * v + q);
                    REQUIRE(cost.Hessian(v) == A);
                    REQUIRE(cost.Hessian() == A);
                    REQUIRE(cost.GetDomainDim() == dim);
                }
            }
        }
    }
}


TEST_CASE("Autodiff Benchmarks", "[autodiff]") {
    using namespace torc::cost;
    using namespace test;
    using adcg_t = CppAD::AD<CppAD::cg::CG<double>>;

    auto fn_ad = functions<adcg_t>;
    const auto tst_dim = 20;
    const auto tst_fn = 6;

    BENCHMARK("Instantiation (No Dynamic Libraries)"){      // ~3e8 ns
        AutodiffCost<double> ad_cost(fn_ad.at(tst_fn), tst_dim, false);
        return ad_cost;
    };

    BENCHMARK("Instantiation (Using Dynamic Libraries)"){   // ~2e4 ns
          AutodiffCost<double> ad_cost(fn_ad.at(tst_fn), tst_dim, true);
          return ad_cost;
    };

    const std::vector<int> test_dims = {1, 50};
    for (auto dim : test_dims) {
        AutodiffCost<double> ad_cost(fn_ad.at(tst_fn), dim, false);     // different dimensions, we avoid the previous
        BENCHMARK("Gradient Evaluation") {
            return ad_cost.Gradient(Eigen::VectorX<double>::Random(dim));
        };
        BENCHMARK("Hessian Evaluation") {
            return ad_cost.Hessian(Eigen::VectorX<double>::Random(dim));
        };
    }
}

TEST_CASE("Differential Consistency Tests", "[analytic][autodiff][finite]") {
    using namespace torc::cost;
    using namespace test;
    using adcg_t = CppAD::AD<CppAD::cg::CG<double>>;

    auto fn_d = functions<double>;
    auto fn_ad = functions<adcg_t>;
    auto grad_d = gradients<double>;
    auto hess_d = hessians<double>;

    // current precision is around 0.003 for Hessians
    const double prec = sqrt(sqrt(std::numeric_limits<double>::epsilon())) * 20;
    const std::vector<int> test_dims = {1, 50};
    const int n_tests = 20;

    for (int i=0; i<fn_d.size(); i++) {
        for (auto dim : test_dims) {
            FiniteDiffCost<double> fd_cost(fn_d.at(i), dim);
            AnalyticCost<double> an_cost(fn_d.at(i), grad_d.at(i), hess_d.at(i), dim);
            AutodiffCost<double> ad_cost(fn_ad.at(i), dim);
            AutodiffCost<double> ad_cost2(fn_ad.at(i), dim, true);  // test load SO
            for (int _=0; _<n_tests; _++){
                Eigen::VectorX<double> input = Eigen::VectorX<double>::Random(dim);
                auto an_eval = an_cost.Evaluate(input);
                auto an_grad = an_cost.Gradient(input);
                auto an_hess = an_cost.Hessian(input);
                auto fd_eval = fd_cost.Evaluate(input);
                auto fd_grad = fd_cost.Gradient(input);
                auto fd_hess = fd_cost.Hessian(input);
                auto ad_eval = ad_cost.Evaluate(input);
                auto ad_grad = ad_cost.Gradient(input);
                auto ad_hess = ad_cost.Hessian(input);
                auto ad_eval2 = ad_cost2.Evaluate(input);
                auto ad_grad2 = ad_cost2.Gradient(input);
                auto ad_hess2 = ad_cost2.Hessian(input);
                REQUIRE_THAT(an_eval, Catch::Matchers::WithinRel(fd_eval, prec));
                REQUIRE_THAT(an_eval, Catch::Matchers::WithinRel(ad_eval, prec));
                REQUIRE_THAT(ad_eval, Catch::Matchers::WithinRel(ad_eval2, prec));
                REQUIRE(an_grad.isApprox(fd_grad, prec));
                REQUIRE(an_grad.isApprox(ad_grad, prec));
                REQUIRE(ad_grad.isApprox(ad_grad2, prec));
                REQUIRE(an_hess.isApprox(fd_hess, prec));
                REQUIRE(an_hess.isApprox(ad_hess, prec));
                REQUIRE(ad_hess2.isApprox(ad_hess2, prec));
            }
        }
    }
}