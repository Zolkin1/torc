//
// Created by zolkin on 8/7/24.
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include <filesystem>

#include "cross_entropy.h"

TEST_CASE("Basic Sample Planner Test", "[sample_planner]") {
    using namespace torc::sample;

    std::filesystem::path achilles_xml = std::filesystem::current_path();
    achilles_xml += "/test_data/achilles.xml";

    CrossEntropy cem(achilles_xml, 10);
}
