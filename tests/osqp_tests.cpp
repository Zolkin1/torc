//
// Created by zolkin on 6/10/24.
//

#include <catch2/catch_test_macros.hpp>

#include "osqp_interface.h"

bool VectorsEqualWithMargin(const torc::solvers::vectorx_t& v1,
                            const torc::solvers::vectorx_t& v2, const double margin) {
    if (v1.size() != v2.size()) {
        return false;
    }

    for (int i = 0; i < v1.size(); i++) {
        if (std::abs(v1(i) - v2(i)) > margin) {
            return false;
        }
    }

    return true;
}

TEST_CASE("Basic OSQPInterface Test", "[optimization][osqp]") {
    using namespace torc::solvers;

    constexpr double MARGIN = 1e-3;

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

    vectorx_t sol(2);

    vectorx_t true_sol(3);
    true_sol << -0.0333, -1.15, -4.0;

    SECTION("Default settings") {
        OSQPInterface qp;

        qp.ResetData(Psparse, w, Asparse, lb, ub);

        SolverStatus status = qp.Solve(sol);
        REQUIRE(status == Solved);
        std::cout << "\nDefault Settings Result: ";
        PrintStatus(std::cout, status);
        REQUIRE(VectorsEqualWithMargin(true_sol, sol, MARGIN));
    }

    SECTION("Changing Settings") {
        OSQPInterface qp;

        qp.ResetData(Psparse, w, Asparse, lb, ub);

        OSQPInterfaceSettings settings;
        settings.verbose = false;
        qp.UpdateSettings(settings);
        SolverStatus status = qp.Solve(sol);
        REQUIRE(status == Solved);
        std::cout << "\nChanging Settings Result: ";
        PrintStatus(std::cout, status);
        REQUIRE(VectorsEqualWithMargin(true_sol, sol, MARGIN));
    }

    SECTION("Initial Settings") {
        OSQPInterfaceSettings settings;
        settings.verbose = false;
        settings.max_iter = 1000;
        OSQPInterface qp(settings);

        qp.ResetData(Psparse, w, Asparse, lb, ub);

        SolverStatus status = qp.Solve(sol);
        REQUIRE(status == Solved);
        std::cout << "\nInitial Settings Result: ";
        PrintStatus(std::cout, status);
        REQUIRE(VectorsEqualWithMargin(true_sol, sol, MARGIN));
    }

    SECTION("Updating the data") {
        OSQPInterfaceSettings settings;
        settings.verbose = false;
        settings.max_iter = 1000;
        OSQPInterface qp(settings);

        qp.ResetData(Psparse, w, Asparse, lb, ub);

        A << 1, 4, 0, 1.3, 0, 0.3;
        Asparse = A.sparseView();

        lb << -2, -7;

        w << 0.1, 4.6, 4;

        P << 6, 0, 0, 0, 2, 0, 0, 0, 0.5;
        Psparse = P.sparseView();

        qp.UpdateData(Psparse, w, Asparse, lb, ub);

        true_sol << 0.1306, -0.5327, -8.0;

        SolverStatus status = qp.Solve(sol);
        REQUIRE(status == Solved);
        std::cout << "\nUpdating Data Result: ";
        PrintStatus(std::cout, status);
        REQUIRE(VectorsEqualWithMargin(true_sol, sol, MARGIN));
    }
}