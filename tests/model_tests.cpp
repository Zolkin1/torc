#include <catch2/catch_test_macros.hpp>

#include "rigid_body.h"

bool VectorEqualWithMargin(const torc::models::vectorx_t& v1, const torc::models::vectorx_t& v2, const double MARGIN) {
    using namespace torc::models;
    if (v1.size() != v2.size()) {
        return false;
    }

    for (int i = 0; i < v1.size(); i++) {
        if (std::abs(v1(i) - v2(i)) > MARGIN) {
            return false;
        }
    }

    return true;
}

TEST_CASE("Quadruped", "[model][pinocchio]") {
    using namespace torc::models;
    const std::string pin_model_name = "test_pin_model";
    const std::string bad_urdf = "fake_urdf.urdf";

    constexpr double MARGIN = 1e-6;

    RobotContactInfo contacts;
    contacts.contacts.emplace("FR_foot", Contact(ContactType::PointContact, false));
    contacts.contacts.emplace("FL_foot", Contact(ContactType::PointContact, false));
    contacts.contacts.emplace("RR_foot", Contact(ContactType::PointContact, false));
    contacts.contacts.emplace("RL_foot", Contact(ContactType::PointContact, false));


    // Check that a bad urdf throws an error
    REQUIRE_THROWS_AS(RigidBody(pin_model_name, bad_urdf, contacts), std::runtime_error);

    std::filesystem::path a1_urdf = std::filesystem::current_path();
    a1_urdf += "/test_data/test_a1.urdf";

    RigidBody pin_model(pin_model_name, a1_urdf, contacts);
    REQUIRE(pin_model.GetName() == pin_model_name);

    constexpr int INPUT_SIZE = 12;
    constexpr int CONFIG_SIZE = 19;
    constexpr int VEL_SIZE = 18;
    constexpr int STATE_SIZE = 37;
    constexpr int DERIV_SIZE = 36;
    constexpr int JOINT_SIZE = 14;
    constexpr int FRAME_SIZE = 47;

    REQUIRE(pin_model.GetNumInputs() == INPUT_SIZE);
    REQUIRE(pin_model.GetConfigDim() == CONFIG_SIZE);
    REQUIRE(pin_model.GetVelDim() == VEL_SIZE);
    REQUIRE(pin_model.GetStateDim() == STATE_SIZE);
    REQUIRE(pin_model.GetDerivativeDim() == DERIV_SIZE);
    REQUIRE(pin_model.GetNumJoints() == JOINT_SIZE);
    REQUIRE(pin_model.GetNumFrames() == FRAME_SIZE);


    // ----------------------------------------------- //
    // ------------------- Dynamics ------------------ //
    // ----------------------------------------------- //
    RobotState x(pin_model.GetConfigDim(), pin_model.GetVelDim());
    const vectorx_t input = vectorx_t::Zero(INPUT_SIZE);

    RobotStateDerivative xdot = pin_model.GetDynamics(x, input);
    RobotStateDerivative true_deriv(VEL_SIZE);
    true_deriv.a(2) = -9.81;
    REQUIRE(VectorEqualWithMargin(xdot.v, true_deriv.v, MARGIN));
    REQUIRE(VectorEqualWithMargin(xdot.a, true_deriv.a, MARGIN));

    // ----------------------------------------------- //
    // ------------- Dynamics Derivatives ------------ //
    // ----------------------------------------------- //
    matrixx_t A, B;
    A = matrixx_t::Zero(pin_model.GetDerivativeDim(), pin_model.GetDerivativeDim());
    B = matrixx_t::Zero(pin_model.GetDerivativeDim(), INPUT_SIZE);

    // TODO: Need input mappings before this is finished
    pin_model.DynamicsDerivative(x, input, A, B);
}

TEST_CASE("Double Integrator", "[model][pinocchio]") {
    using namespace torc::models;
    const std::string pin_model_name = "test_pin_model";

    std::filesystem::path integrator_urdf = std::filesystem::current_path();
    integrator_urdf += "/test_data/integrator.urdf";

    // No contacts
    RobotContactInfo contacts;

    RigidBody pin_model(pin_model_name, integrator_urdf, contacts);

    REQUIRE(pin_model.GetMass() == 1);
}

TEST_CASE("Hopper", "[model][pinocchio]") {
    using namespace torc::models;
    const std::string pin_model_name = "test_pin_model";

    std::filesystem::path hopper_urdf = std::filesystem::current_path();
    hopper_urdf += "/test_data/hopper.urdf";

    RobotContactInfo contacts;
    contacts.contacts.emplace("foot", Contact(PointContact, false));

    RigidBody pin_model(pin_model_name, hopper_urdf, contacts);

    int constexpr INPUT_SIZE = 4;
    constexpr int CONFIG_SIZE = 11;
    constexpr int VEL_SIZE = 10;
    constexpr int STATE_SIZE = 21;
    constexpr int DERIV_SIZE = 20;
    constexpr int JOINT_SIZE = 6;

    REQUIRE(pin_model.GetNumInputs() == INPUT_SIZE);
    REQUIRE(pin_model.GetConfigDim() == CONFIG_SIZE);
    REQUIRE(pin_model.GetVelDim() == VEL_SIZE);
    REQUIRE(pin_model.GetStateDim() == STATE_SIZE);
    REQUIRE(pin_model.GetDerivativeDim() == DERIV_SIZE);
    REQUIRE(pin_model.GetNumJoints() == JOINT_SIZE);
}