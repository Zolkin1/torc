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
}