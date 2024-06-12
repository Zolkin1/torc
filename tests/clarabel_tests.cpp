//
// Created by zolkin on 6/12/24.
//

#include <catch2/catch_test_macros.hpp>

#include "clarabel_interface.h"

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

TEST_CASE("Basic Clarabel Test", "[optimization][clarabel]") {
    using namespace torc::solvers;

    constexpr double MARGIN = 1e-4;

    matrixx_t A(4,3);
    A << 1, 1, 0, 1.3, 0, 0.2,
         -1, -1, 0, -1.3, 0, -0.2;
    sp_matrixx_t Asparse = A.sparseView();

    vectorx_t lb(2);
    lb << -2, -5;

    vectorx_t ub(2);
    ub << 3.1, 13.3;

    vectorx_t b(4);
    b << ub, -lb;

    matrixx_t P(3,3);
    P << 3.001, 0, 0, 0, 4, 0, 0, 0, 0.5;
    sp_matrixx_t Psparse = P.sparseView();

    vectorx_t w(3);
    w << 0.1, 4.6, 2;

    vectorx_t sol(2);

    vectorx_t true_sol(3);
    true_sol << -0.0333, -1.15, -4.0;

    ClarabelData data;
    data.Q = Psparse;
    data.q = w;
    data.A = Asparse;
    data.b = b;
    data.constraint_data.emplace_back(UpperBound, 2);
    data.constraint_data.emplace_back(UpperBound, 2);

    SECTION("Default settings") {
        ClarabelInterface qp;

        qp.SetData(data);

        SolverStatus status = qp.Solve(sol);

        REQUIRE(status == Solved);
        std::cout << "\nDefault Settings Result: ";
        PrintStatus(std::cout, status);
        REQUIRE(VectorsEqualWithMargin(true_sol, sol, MARGIN));
    }

    SECTION("Changing Settings After Data") {
        ClarabelInterface qp;

        qp.SetData(data);

        // update settings after SetData
        ClarabelInterfaceSettings settings;
        settings.verbose = false;
        settings.max_iters = 11;
        qp.UpdateSettings(settings);

        SolverStatus status = qp.Solve(sol);

        REQUIRE(status == Solved);
        std::cout << "\nChanging Settings Result: ";
        PrintStatus(std::cout, status);
        REQUIRE(VectorsEqualWithMargin(true_sol, sol, MARGIN));
    }

    SECTION("Changing Settings Before Data") {
        ClarabelInterface qp;

        // update settings after SetData
        ClarabelInterfaceSettings settings;
        settings.verbose = false;
        settings.max_iters = 1;
        qp.UpdateSettings(settings);

        REQUIRE(qp.GetSettings() == settings);

        qp.SetData(data);

        SolverStatus status = qp.Solve(sol);

        REQUIRE(status == MaxIters);
        std::cout << "\nChanging Settings Result: ";
        PrintStatus(std::cout, status);
    }


    SECTION("Settings In Constructor") {
        // update settings after SetData
        ClarabelInterfaceSettings settings;
        settings.verbose = false;
        settings.max_iters = 1;

        ClarabelInterface qp(settings);

        qp.SetData(data);

        REQUIRE(qp.GetSettings() == settings);

        SolverStatus status = qp.Solve(sol);

        REQUIRE(status == MaxIters);
        std::cout << "\nChanging Settings Result: ";
        PrintStatus(std::cout, status);
    }
}