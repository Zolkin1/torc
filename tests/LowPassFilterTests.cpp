//
// Created by zolkin on 2/15/25.
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "low_pass_filter.h"

#define ENABLE_BENCHMARKS false

TEST_CASE("LPF Test", "[LPF]") {
    using namespace torc::state_est;
    std::vector<double> coefs = {0.25, 0.75};

    torc::state_est::LowPassFilter lpf(coefs);

    vectorx_t x1 = vectorx_t::Ones(5);
    vectorx_t res1 = lpf.Filter(x1);
    CHECK(res1.isApprox(x1*coefs[0]));

    vectorx_t x2 = 3*vectorx_t::Ones(5);
    vectorx_t res2 = lpf.Filter(x2);
    CHECK(res2.isApprox(x1*coefs[0] + x2*coefs[1]));

    vectorx_t x3 = 2.1*vectorx_t::Ones(5);
    vectorx_t res3 = lpf.Filter(x3);
    CHECK(res3.isApprox(x2*coefs[0] + x3*coefs[1]));
}