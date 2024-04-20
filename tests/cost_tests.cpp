#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <eigen3/Eigen/Dense>
#include "../costs/linear_cost.h"
#include <cstdint>


TEST_CASE("Linear Cost Test") {
    Eigen::Vector3d v, x1;
    v << 1, 2, 3;
    x1 << 1, 1, 1;
    std::string s = "hi";
    torc::LinearCost cost = torc::LinearCost<double>(v, s);
    REQUIRE(cost.Evaluate(x1) == 6);
}