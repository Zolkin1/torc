//
// Created by zolkin on 8/7/24.
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "cross_entropy.h"

TEST_CASE("Basic Sample Planner Test", "[sample_planner]") {
    using namespace torc::sample;
    CrossEntropy cem();
}
