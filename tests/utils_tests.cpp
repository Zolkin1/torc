//
// Created by zolkin on 6/11/24.
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <thread>
#include <iostream>

#include "torc_timer.h"

TEST_CASE("Timer Test", "[timer][utils]") {
    using namespace torc::utils;
    using Catch::Matchers::WithinAbs;

    constexpr double MARGIN = 1e-8;
    constexpr int MS_DELAY = 100;

    TORCTimer timer;
    timer.Tic();
    std::this_thread::sleep_for(std::chrono::milliseconds(MS_DELAY));
    timer.Toc();

    std::cout << "measured (s): " << timer.Duration<std::chrono::seconds >().count() << std::endl;
    std::cout << "measured (ms): " << timer.Duration<std::chrono::milliseconds>().count() << std::endl;
    std::cout << "measured (us): " << timer.Duration<std::chrono::microseconds>().count() << std::endl;
    std::cout << "measured (ns): " << timer.Duration<std::chrono::nanoseconds>().count() << std::endl;

    REQUIRE_THAT(timer.Duration<std::chrono::milliseconds>().count() - MS_DELAY, WithinAbs(0, MARGIN));
    REQUIRE_THAT(timer.Duration<std::chrono::microseconds>().count() - MS_DELAY*1000, WithinAbs(0, 500));
    REQUIRE_THAT(timer.Duration<std::chrono::nanoseconds>().count() - MS_DELAY*1000000, WithinAbs(0, 500000));
    REQUIRE_THAT(timer.Duration<std::chrono::seconds>().count() - MS_DELAY/100, WithinAbs(0, 1));
}