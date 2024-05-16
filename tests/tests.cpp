#define CATCH_CONFIG_MAIN

#include <random>
#include <iostream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/benchmark/catch_constructor.hpp>
#include <eigen3/Eigen/Dense>
#include <cppad/cg.hpp>

#include "autodiff_fn.h"
#include "test_fn.h"


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
        //FiniteDiffFn<double> fd_fn(fn_d.at(i), dim);
            ExplicitFn<double> an_fn(fn_d.at(i), grad_d.at(i), hess_d.at(i), dim);
            AutodiffFn<double> ad_fn(fn_ad.at(i), dim, true);
            AutodiffFn<double> ad_fn2(fn_ad.at(i), dim, false);  // test load library implicitly
            AutodiffFn<double> ad_fn3(fn_ad.at(i), "./adcg_sources/AutodiffFnInstance.so");  // test load library from filename
            for (int _=0; _<n_tests; _++) {
                Eigen::VectorX<double> input = Eigen::VectorX<double>::Random(dim);
                auto ad_eval = ad_fn.Evaluate(input);
                auto ad_grad = ad_fn.Gradient(input);
                auto ad_hess = ad_fn.Hessian(input);
                auto ad_eval2 = ad_fn2.Evaluate(input);
                auto ad_grad2 = ad_fn2.Gradient(input);
                auto ad_hess2 = ad_fn2.Hessian(input);
                auto ad_eval3 = ad_fn3.Evaluate(input);
                auto ad_grad3 = ad_fn3.Gradient(input);
                auto ad_hess3 = ad_fn3.Hessian(input);
                REQUIRE_THAT(ad_eval, Catch::Matchers::WithinRel(ad_eval2, 1e-7));
                REQUIRE_THAT(ad_eval, Catch::Matchers::WithinRel(ad_eval3, 1e-7));
                REQUIRE(ad_grad.isApprox(ad_grad2, prec));
                REQUIRE(ad_grad.isApprox(ad_grad3, prec));
                REQUIRE(ad_hess2.isApprox(ad_hess2, prec));
                REQUIRE(ad_hess2.isApprox(ad_hess3, prec));
            }
        }
    }
}