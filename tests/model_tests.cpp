#include <catch2/catch_test_macros.hpp>

#include "rigid_body.h"

//TEST_CASE("Basic Model", "[model]") {
//    using namespace torc::models;
//    const std::string base_model_name = "test_base_model";
//    BaseModel base_model(base_model_name);
//
//    REQUIRE(base_model.GetName() == base_model_name);
//}

TEST_CASE("Basic Pinocchio Model", "[model][pinocchio]") {
    using namespace torc::models;
    const std::string pin_model_name = "test_pin_model";
    const std::string bad_urdf = "fake_urdf.urdf";

    // Check that a bad urdf throws an error
    REQUIRE_THROWS_AS(RigidBody(pin_model_name, bad_urdf), std::runtime_error);

    std::filesystem::path a1_urdf = std::filesystem::current_path();
    a1_urdf += "/test_data/test_a1.urdf";

    RigidBody pin_model(pin_model_name, a1_urdf);
    REQUIRE(pin_model.GetName() == pin_model_name);
}

TEST_CASE("Double Integrator", "[model][pinocchio]") {
    using namespace torc::models;
    const std::string pin_model_name = "test_pin_model";

    std::filesystem::path integrator_urdf = std::filesystem::current_path();
    integrator_urdf += "/test_data/integrator.urdf";

    RigidBody pin_model(pin_model_name, integrator_urdf);

    REQUIRE(pin_model.GetMass() == 1);

    RobotState x(pin_model.GetConfigDim(), pin_model.GetVelDim());
    const vectorx_t input = vectorx_t::Zero(3);

    RobotStateDerivative xdot = pin_model.GetDynamics(x, input);
    RobotStateDerivative true_deriv(6);
    true_deriv.a(2) = -9.81;
    REQUIRE(xdot == true_deriv);
}