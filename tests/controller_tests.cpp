//
// Created by zolkin on 7/18/24.
//
#include <catch2/catch_test_macros.hpp>

#include "whole_body_qp_controller.h"

TEST_CASE("Basic WBC Tests", "[contollers][wbc]") {
    using namespace torc::controllers;
    WholeBodyQPController();
}