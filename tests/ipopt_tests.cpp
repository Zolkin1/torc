//
// Created by zolkin on 6/7/24.
//

#include <catch2/catch_test_macros.hpp>

#include "ipopt.h"

TEST_CASE("Basic IPOPT Test", "[optimization][ipopt]") {
    using namespace torc::solvers;

    IPOPT nlp;
    SolverStatus status = nlp.SolveNLP();
    PrintStatus(std::cout, status);
}