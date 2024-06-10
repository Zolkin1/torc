//
// Created by zolkin on 6/10/24.
//

#include <catch2/catch_test_macros.hpp>

#include "osqp_interface.h"

TEST_CASE("Basic OSQPInterface Test", "[optimization][osqp]") {
    using namespace torc::solvers;

    OSQPInterface qp;

    matrixx_t A(2,3);
    A << 1, 1, 0, 1.3, 0, 0.2;
    sp_matrixx_t Asparse = A.sparseView();

    vectorx_t lb(2);
    lb << -2, -5;

    vectorx_t ub(2);
    ub << 3.1, 13.3;

    matrixx_t P(3,3);
    P << 3.001, 0, 0, 0, 4, 0, 0, 0, 0.5;
    sp_matrixx_t Psparse = P.sparseView();

    vectorx_t w(3);
    w << 0.1, 4.6, 2;

    qp.ResetData(Psparse, w, Asparse, lb, ub);

    vectorx_t sol(2);

    SolverStatus status = qp.Solve(sol);
    PrintStatus(std::cout, status);
}