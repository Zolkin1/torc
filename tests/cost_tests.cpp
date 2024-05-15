#define CATCH_CONFIG_MAIN

#include <random>
#include <iostream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/benchmark/catch_constructor.hpp>
#include <eigen3/Eigen/Dense>
#include <cppad/cg.hpp>

#include "linear_fn.h"
#include "quadratic_fn.h"
#include "autodiff_fn.h"
#include "analytic_fn.h"
#include "finite_diff_fn.h"
#include "test_fn.h"


TEST_CASE("Linear Function Test", "[linear]") {
    Eigen::Vector3d q1, x1, x2;
    q1 << 1, 2, 3;
    x1 << 0.1, 1, 10;
    x2 << 0, 0, 0;
    std::string name1 = "LinearFn1";
    torc::fn::LinearFn lin1 = torc::fn::LinearFn<double>(q1, name1);
    REQUIRE(lin1.Evaluate(x1) == 32.1);
    REQUIRE(lin1.GetDomainDim() == 3);
    REQUIRE(lin1.GetCoefficients() == q1);
    REQUIRE(lin1.Gradient() == q1);
    REQUIRE(lin1.Gradient(x2) == q1);
    REQUIRE(lin1.Hessian(x1) == Eigen::MatrixXd::Zero(3, 3));
    REQUIRE(lin1.GetIdentifier() == name1);

    Eigen::Vector2f q2, x3;
    q2 << 1.5, 2.5;
    x3 << 4, 8;
    std::string name2 = "LinearFn1";
    torc::fn::LinearFn lin2 = torc::fn::LinearFn<float>(q2, name2);
    REQUIRE(lin2.Evaluate(x3) == 26);
    REQUIRE(lin2.Gradient() == q2);
}


TEST_CASE("Quadratic Function Test", "[quadratic]") {
    using namespace torc::fn;
    const int n_tests = 10;
    const std::vector<int> test_dims = {1, 50};

    SECTION("Full matrix") {
        for (int i=0; i<n_tests; i++) {
            for (auto dim : test_dims) {
                Eigen::MatrixX<double> A = Eigen::MatrixX<double>::Random(dim, dim).selfadjointView<Eigen::Upper>();
                torc::fn::QuadraticFn<double> quad = torc::fn::QuadraticFn(A);
                for (int j = 0; j < n_tests; ++j) {
                    Eigen::VectorX<double> v = Eigen::VectorX<double>::Random(dim);
                    REQUIRE(quad.Evaluate(v) == 0.5 * (v.dot(A * v)));
                    REQUIRE(quad.GetQuadCoefficients() == A);
                    REQUIRE(quad.GetLinCoefficients() == Eigen::VectorX<double>::Zero(dim));
                    REQUIRE(quad.Gradient(v) == A * v);
                    REQUIRE(quad.Hessian(v) == A);
                    REQUIRE(quad.GetDomainDim() == dim);
                }
            }
        }
    }

    SECTION("Triangular View") {    // matrix has to be fixed-size at compile time to call triangularView
        for (int i=0; i<n_tests; i++) {
            Eigen::Matrix3d A = Eigen::Matrix3d::Random();
            Eigen::TriangularView<Eigen::Matrix3d, Eigen::Upper> Au = A.triangularView<Eigen::Upper>();
            torc::fn::QuadraticFn<double> quad = torc::fn::QuadraticFn(Au);
            A = A.selfadjointView<Eigen::Upper>();
            for (int j = 0; j < n_tests; ++j) {
                Eigen::Vector3d v = Eigen::Vector3d::Random();
                REQUIRE(abs(quad.Evaluate(v) - 0.5 * (v.dot(A * v))) < 1e-8);
                REQUIRE(quad.GetQuadCoefficients() == A);
                REQUIRE(quad.GetLinCoefficients() == Eigen::Vector3d::Zero());
                REQUIRE(quad.Gradient(v).isApprox(A * v));
                REQUIRE(quad.Hessian(v).isApprox(A));
                REQUIRE(quad.GetDomainDim() == 3);
            }
        }
    }

    SECTION("Full matrix with linear fn") {
        for (int i=0; i<n_tests; i++) {
            for (auto dim: test_dims) {
                Eigen::MatrixX<double> A = Eigen::MatrixX<double>::Random(dim, dim).selfadjointView<Eigen::Upper>();
                Eigen::VectorX<double> q = Eigen::VectorX<double>::Random(dim);
                torc::fn::QuadraticFn<double> quad = torc::fn::QuadraticFn(A, q);
                for (int j = 0; j < n_tests; ++j) {
                    Eigen::VectorX<double> v = Eigen::VectorX<double>::Random(dim);
                    REQUIRE(quad.Evaluate(v) == 0.5 * (v.dot(A * v)) + q.dot(v));
                    REQUIRE(quad.GetQuadCoefficients() == A);
                    REQUIRE(quad.GetLinCoefficients() == q);
                    REQUIRE(quad.Gradient(v) == A * v + q);
                    REQUIRE(quad.Hessian(v) == A);
                    REQUIRE(quad.Hessian() == A);
                    REQUIRE(quad.GetDomainDim() == dim);
                }
            }
        }
    }
}


TEST_CASE("Autodiff Benchmarks", "[autodiff]") {
    using namespace torc::fn;
    using namespace test;
    using adcg_t = CppAD::AD<CppAD::cg::CG<double>>;

    auto fn_ad = functions<adcg_t>;
    const auto tst_dim = 20;
    const auto tst_fn = 6;

    auto fn = fn_ad.at(tst_fn);

    BENCHMARK("Instantiation (No Dynamic Libraries)"){      // ~3e8 ns
        AutodiffFn<double> ad_fn(fn, tst_dim, true, false);
        return ad_fn;
    };
    BENCHMARK("Instantiation (Automatically Using Dynamic Libraries)"){   // ~2e4 ns
        AutodiffFn<double> ad_fn(fn, tst_dim, false, false);
        return ad_fn;
    };
    BENCHMARK("Instantiation (Manually Using Dynamic Libraries)"){   // ~2e4 ns
        AutodiffFn<double> ad_fn(fn, "./adcg_sources/AutodiffFnInstance.so");
        return ad_fn;
    };

    const std::vector<int> test_dims = {1, 50};
    for (auto dim : test_dims) {
        AutodiffFn<double> ad_fn(fn_ad.at(tst_fn), dim, true);     // different dimensions, we avoid the previous
        BENCHMARK("Gradient Evaluation") {
            return ad_fn.Gradient(Eigen::VectorX<double>::Random(dim));
        };
        BENCHMARK("Hessian Evaluation") {
            return ad_fn.Hessian(Eigen::VectorX<double>::Random(dim));
        };
    }
}


TEST_CASE("Differential Consistency Tests", "[analytic][autodiff][finite]") {
    using namespace torc::fn;
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
            FiniteDiffFn<double> fd_fn(fn_d.at(i), dim);
            AnalyticalFn<double> an_fn(fn_d.at(i), grad_d.at(i), hess_d.at(i), dim);
            AutodiffFn<double> ad_fn(fn_ad.at(i), dim, true);
            AutodiffFn<double> ad_fn2(fn_ad.at(i), dim, false);  // test load library implicitly
            AutodiffFn<double> ad_fn3(fn_ad.at(i), "./adcg_sources/AutodiffFnInstance.so");  // test load library from filename
            for (int _=0; _<n_tests; _++){
                Eigen::VectorX<double> input = Eigen::VectorX<double>::Random(dim);
                auto an_eval = an_fn.Evaluate(input);
                auto an_grad = an_fn.Gradient(input);
                auto an_hess = an_fn.Hessian(input);
                auto fd_eval = fd_fn.Evaluate(input);
                auto fd_grad = fd_fn.Gradient(input);
                auto fd_hess = fd_fn.Hessian(input);
                auto ad_eval = ad_fn.Evaluate(input);
                auto ad_grad = ad_fn.Gradient(input);
                auto ad_hess = ad_fn.Hessian(input);
                auto ad_eval2 = ad_fn2.Evaluate(input);
                auto ad_grad2 = ad_fn2.Gradient(input);
                auto ad_hess2 = ad_fn2.Hessian(input);
                auto ad_eval3 = ad_fn3.Evaluate(input);
                auto ad_grad3 = ad_fn3.Gradient(input);
                auto ad_hess3 = ad_fn3.Hessian(input);
                REQUIRE_THAT(an_eval, Catch::Matchers::WithinRel(fd_eval, prec));
                REQUIRE_THAT(an_eval, Catch::Matchers::WithinRel(ad_eval, prec));
                REQUIRE_THAT(ad_eval, Catch::Matchers::WithinRel(ad_eval2, prec));
                REQUIRE_THAT(ad_eval, Catch::Matchers::WithinRel(ad_eval3, prec));
                REQUIRE(an_grad.isApprox(fd_grad, prec));
                REQUIRE(an_grad.isApprox(ad_grad, prec));
                REQUIRE(ad_grad.isApprox(ad_grad2, prec));
                REQUIRE(ad_grad.isApprox(ad_grad3, prec));
                REQUIRE(an_hess.isApprox(fd_hess, prec));
                REQUIRE(an_hess.isApprox(ad_hess, prec));
                REQUIRE(ad_hess2.isApprox(ad_hess2, prec));
                REQUIRE(ad_hess2.isApprox(ad_hess3, prec));
            }
        }
    }
}