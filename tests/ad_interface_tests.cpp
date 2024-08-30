//
// Created by zolkin on 8/21/24.
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "auto_diff_types.h"
#include "cpp_ad_interface.h"

void TestFunction(const torc::ad::ad_vector_t& x, const torc::ad::ad_vector_t& p, torc::ad::ad_vector_t& y) {
    y = p(0)*x;
}

void PowTestFunction(const torc::ad::ad_vector_t& x, const torc::ad::ad_vector_t& p, torc::ad::ad_vector_t& y) {
    y.resize(2);
    y(0) = CppAD::pow(x(0), 2);
    y(1) = CppAD::pow(p(0)*x(0), 2);
}

TEST_CASE("Basic AD Interface Tests", "[ad]") {
    using namespace torc::ad;

    int constexpr X_SIZE = 5;
    int constexpr P_SIZE = 2;
    int constexpr Y_SIZE = 5;

    auto curr_path = fs::current_path();
    curr_path = curr_path / "deriv_libs";
    CppADInterface function(&TestFunction, "test_ad_function", curr_path, SecondOrder, X_SIZE, P_SIZE, true);

    CHECK(function.GetDomainSize() == X_SIZE);
    CHECK(function.GetParameterSize() == P_SIZE);
    CHECK(function.GetRangeSize() == Y_SIZE);

    // Check the function value
    vectorx_t p  = vectorx_t::Random(P_SIZE);
    vectorx_t x = vectorx_t::Random(X_SIZE);

    vectorx_t y = p(0)*x;
    vectorx_t y_test;
    function.GetFunctionValue(x, p, y_test);
    CHECK(y_test.size() == function.GetRangeSize());
    CHECK(y_test == y);


    // Check the jacobian
    matrixx_t jac;
    p  = vectorx_t::Random(P_SIZE);
    function.GetJacobian(vectorx_t::Random(X_SIZE), p, jac);

    CHECK(jac.rows() == function.GetRangeSize());
    CHECK(jac.cols() == function.GetDomainSize());

    CHECK(jac == p(0)*matrixx_t::Identity(Y_SIZE, X_SIZE));

    // Check the hessian
    matrixx_t hess;
    function.GetHessian(vectorx_t::Random(X_SIZE), vectorx_t::Random(P_SIZE), vectorx_t::Ones(X_SIZE), hess);

    CHECK(hess == matrixx_t::Zero(X_SIZE, X_SIZE));

    // Check gauss newton
    p = vectorx_t::Random(P_SIZE);
    function.GetGaussNewton(vectorx_t::Random(X_SIZE), p, jac, hess);
    CHECK(jac == p(0)*matrixx_t::Identity(Y_SIZE, X_SIZE));
    CHECK(hess == jac.transpose() * jac);

    // Check sparsity patterns
    jac.setZero();
    function.GetJacobianSparsityPatternMat(jac);
    CHECK(jac == matrixx_t::Identity(Y_SIZE, X_SIZE));

    hess.setZero();
    function.GetHessianSparsityPatternMat(hess);
    CHECK(hess.isZero());

    hess.setZero();
    function.GetGaussNewtonSparsityPatternMat(hess);
    CHECK(hess == matrixx_t::Identity(X_SIZE, X_SIZE));
}

TEST_CASE("Pow Test", "[ad]") {
    using namespace torc::ad;

    double constexpr MARGIN = 1e-8;

    int constexpr X_SIZE = 1;
    int constexpr P_SIZE = 1;
    int constexpr Y_SIZE = 2;

    auto curr_path = fs::current_path();
    curr_path = curr_path / "deriv_libs";
    CppADInterface function(&PowTestFunction, "pow_test_ad_function", curr_path, FirstOrder, X_SIZE, P_SIZE, true);

    CHECK(function.GetDomainSize() == X_SIZE);
    CHECK(function.GetParameterSize() == P_SIZE);
    CHECK(function.GetRangeSize() == Y_SIZE);

    // Check the function value
    vectorx_t p  = vectorx_t::Random(P_SIZE);
    vectorx_t x = vectorx_t::Random(X_SIZE);

    vectorx_t y(Y_SIZE);
    y << x(0)*x(0), std::pow(p(0)*x(0),2);

    vectorx_t y_test;
    function.GetFunctionValue(x, p, y_test);
    CHECK(y_test.size() == function.GetRangeSize());
    CHECK(y_test == y);

    // Check the jacobian
    matrixx_t jac;
    p  = vectorx_t::Random(P_SIZE);
    x = vectorx_t::Random(X_SIZE);
    function.GetJacobian(x, p, jac);
    CHECK(jac.rows() == function.GetRangeSize());
    CHECK(jac.cols() == function.GetDomainSize());

    matrixx_t jac_analytic(Y_SIZE, X_SIZE);
    jac_analytic(0, 0) = 2*x(0);
    jac_analytic(1, 0) = 2*p(0)*p(0)*x(0);

    CHECK(jac.isApprox(jac_analytic, MARGIN));
}

TEST_CASE("Sparsity", "[ad]") {
    using namespace torc::ad;

    double constexpr MARGIN = 1e-8;

    int constexpr X_SIZE = 1;
    int constexpr P_SIZE = 1;
    int constexpr Y_SIZE = 2;

    auto curr_path = fs::current_path();
    curr_path = curr_path / "deriv_libs";
    CppADInterface function(&PowTestFunction, "sparsity_test_ad_function", curr_path, SecondOrder, X_SIZE, P_SIZE, true);

    CHECK(function.GetDomainSize() == X_SIZE);
    CHECK(function.GetParameterSize() == P_SIZE);
    CHECK(function.GetRangeSize() == Y_SIZE);

    // Get the sparsity pattern
    const auto sp_set = function.GetHessianSparsityPatternSet();
    matrixx_t sp_mat;
    function.GetHessianSparsityPatternMat(sp_mat);

    // Get the hessian
    vectorx_t p  = vectorx_t::Random(P_SIZE);
    vectorx_t x = vectorx_t::Random(X_SIZE);

    matrixx_t hess;
    vectorx_t w = vectorx_t::Ones(Y_SIZE);
    function.GetHessian(x, p, w, hess);

     // Confirm sparsity pattern
     for (int row = 0 ; row < hess.rows(); ++row) {
        for (int col = 0 ; col < hess.cols(); ++col) {
            if (hess(row, col) != 0) {
                CHECK(sp_mat(row, col) == 1);
                CHECK(sp_set.at(row).contains(col));
            }
        }
     }
}

TEST_CASE("Loading", "[ad]") {
    using namespace torc::ad;

    double constexpr MARGIN = 1e-8;

    int constexpr X_SIZE = 1;
    int constexpr P_SIZE = 1;
    int constexpr Y_SIZE = 2;

    auto curr_path = fs::current_path();
    curr_path = curr_path / "deriv_libs";
    CppADInterface function(&PowTestFunction, "pow_test_load_ad_function", curr_path, FirstOrder, X_SIZE, P_SIZE, true);

    CHECK(function.GetDomainSize() == X_SIZE);
    CHECK(function.GetParameterSize() == P_SIZE);
    CHECK(function.GetRangeSize() == Y_SIZE);

    // TODO: Fix!
    // Load
    CppADInterface function2(&PowTestFunction, "pow_test_load_ad_function", curr_path, FirstOrder, X_SIZE, P_SIZE, false);

    CHECK(function2.GetDomainSize() == X_SIZE);
    CHECK(function2.GetParameterSize() == P_SIZE);
    CHECK(function2.GetRangeSize() == Y_SIZE);

    const auto load_sp_jac = function2.GetJacobianSparsityPatternSet();
    const auto sp_jac = function.GetJacobianSparsityPatternSet();

    CHECK(load_sp_jac == sp_jac);
}