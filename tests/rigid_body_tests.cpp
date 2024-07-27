#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "pinocchio/algorithm/aba.hpp"

#include "full_order_rigid_body.h"

#include "full_order_test_fcns.h"

TEST_CASE("Quadruped", "[model][pinocchio]") {
    using namespace torc::models;
    const std::string pin_model_name = "test_pin_model";
    const std::string bad_urdf = "fake_urdf.urdf";

    constexpr double MARGIN = 1e-6;

    // Check that a bad urdf throws an error
    REQUIRE_THROWS_AS(FullOrderRigidBody(pin_model_name, bad_urdf), std::runtime_error);

    RobotContactInfo contact_info;
    contact_info.contacts.emplace("FR_foot", Contact(ContactType::PointContact));
    contact_info.contacts.emplace("FL_foot", Contact(ContactType::PointContact, false));
    contact_info.contacts.emplace("RR_foot", Contact(ContactType::PointContact, false));
    contact_info.contacts.emplace("RL_foot", Contact(ContactType::PointContact));

    std::filesystem::path a1_urdf = std::filesystem::current_path();
    a1_urdf += "/test_data/test_a1.urdf";

    FullOrderRigidBody a1_model(pin_model_name, a1_urdf);
    REQUIRE(a1_model.GetName() == pin_model_name);

    constexpr int INPUT_SIZE = 12;
    constexpr int CONFIG_SIZE = 19;
    constexpr int VEL_SIZE = 18;
    constexpr int STATE_SIZE = 37;
    constexpr int DERIV_SIZE = 36;
    constexpr int JOINT_SIZE = 14;
    constexpr int FRAME_SIZE = 47;

    REQUIRE(a1_model.GetNumInputs() == INPUT_SIZE);
    REQUIRE(a1_model.GetConfigDim() == CONFIG_SIZE);
    REQUIRE(a1_model.GetVelDim() == VEL_SIZE);
    REQUIRE(a1_model.GetStateDim() == STATE_SIZE);
    REQUIRE(a1_model.GetDerivativeDim() == DERIV_SIZE);
    REQUIRE(a1_model.GetNumJoints() == JOINT_SIZE);
    REQUIRE(a1_model.GetNumFrames() == FRAME_SIZE);
    REQUIRE(a1_model.GetSystemType() == HybridSystemImpulse);


    // ----------------------------------------------- //
    // ------------------- Dynamics ------------------ //
    // ----------------------------------------------- //
    vectorx_t x = vectorx_t::Zero(STATE_SIZE);
    const vectorx_t input = vectorx_t::Zero(INPUT_SIZE);

    // No contacts
    vectorx_t xdot = a1_model.GetDynamics(x, input);
    vectorx_t true_deriv = vectorx_t::Zero(DERIV_SIZE);
    vectorx_t a, v;
    a1_model.ParseStateDerivative(true_deriv, v, a);
    a(2) = -9.81;
    true_deriv = FullOrderRigidBody::BuildState(v, a);

    REQUIRE(VectorEqualWithMargin(xdot, true_deriv, MARGIN));

    // Impulse dynamics
    x = a1_model.GetImpulseDynamics(x, input, contact_info);

    // ----------------------------------------------- //
    // ------------- Dynamics Derivatives ------------ //
    // ----------------------------------------------- //
    matrixx_t A, B, Aimp, Bimp;
    A = matrixx_t::Zero(a1_model.GetDerivativeDim(), a1_model.GetDerivativeDim());
    B = matrixx_t::Zero(a1_model.GetDerivativeDim(), INPUT_SIZE);
    Aimp = A;
    Bimp = B;

//    bool dyn_deriv = GENERATE(true, false);
//    if (dyn_deriv) {
//        // No contact dynamics derivatives
//        a1_model.DynamicsDerivative(x, input, A, B);
//    }
//
//    bool contact_deriv = GENERATE(true, false);
//    if (contact_deriv) {
//        // Contact dynamics derivatives
//        a1_model.DynamicsDerivative(x, input, contact_info, A, B);
//    }
//
//    bool impulse_deriv = GENERATE(true, false);
//    if (impulse_deriv) {
//        // Impulse derivative
//        a1_model.ImpulseDerivative(x, input, contact_info, Aimp, Bimp);
//    }
//
//    bool deriv_sec = GENERATE(true, false);
//    if (deriv_sec) {
//        SECTION("Dynamics Derivatives") {
//            // TODOl: Only works when ImpulseDerivative is called above this
//            CheckDerivatives(a1_model);
//        }
//    }
//
//    bool contact_sec = GENERATE(true, false);
//    if (contact_sec) {
//        SECTION("Contact Dynamics Derivatives") {
//            CheckContactDerivatives(a1_model, contact_info);
//        }
//    }
//
//    bool impulse_sec = GENERATE(true, false);
//    if (impulse_sec) {
//        SECTION("Impulse Dynamics Derivatives") {
//            CheckImpulseDerivatives(a1_model, contact_info);
//
//            contact_info.contacts.at("FR_foot").state = true;
//            CheckImpulseDerivatives(a1_model, contact_info);
//        }
//    }
}

//TEST_CASE("Quadruped Benchmarks", "[model][pinocchio][benchmarks]") {
//    using namespace torc::models;
//
//    std::filesystem::path a1_urdf = std::filesystem::current_path();
//    a1_urdf += "/test_data/test_a1.urdf";
//
//    FullOrderRigidBody a1_model("a1", a1_urdf);
//
//    matrixx_t A, B, Aimp, Bimp;
//    A = matrixx_t::Zero(a1_model.GetDerivativeDim(), a1_model.GetDerivativeDim());
//    B = matrixx_t::Zero(a1_model.GetDerivativeDim(), a1_model.GetNumInputs());
//
//    RobotContactInfo contact_info;
//    contact_info.contacts.emplace("FR_foot", Contact(ContactType::PointContact));
//    contact_info.contacts.emplace("FL_foot", Contact(ContactType::PointContact, true));
//    contact_info.contacts.emplace("RR_foot", Contact(ContactType::PointContact, true));
//    contact_info.contacts.emplace("RL_foot", Contact(ContactType::PointContact));
//
//    vectorx_t x_rand = a1_model.GetRandomState();
//    vectorx_t input_rand = vectorx_t::Random(a1_model.GetNumInputs());
//    // Benchmarks
//    BENCHMARK("Dynamics Derivatives") {
//        return a1_model.DynamicsDerivative(x_rand, input_rand, A, B);
//    };
//
//    BENCHMARK("Contact Dynamics Derivatives") {
//        return a1_model.DynamicsDerivative(x_rand, input_rand, contact_info, A, B);
//    };
//
//    BENCHMARK("Impulse Dynamics Derivatives") {
//        return a1_model.ImpulseDerivative(x_rand, input_rand, contact_info, A, B);
//    };
//
//    BENCHMARK("Dynamics") {
//        return a1_model.GetDynamics(x_rand, input_rand);
//    };
//
//    BENCHMARK("Contact Dynamics") {
//        return a1_model.GetDynamics(x_rand, input_rand, contact_info);
//    };
//
//    BENCHMARK("Impulse Dynamics") {
//        return a1_model.GetImpulseDynamics(x_rand, input_rand, contact_info);
//    };
//}

TEST_CASE("Double Integrator", "[model][pinocchio]") {
    using namespace torc::models;
    const std::string pin_model_name = "test_pin_model";

    std::filesystem::path integrator_urdf = std::filesystem::current_path();
    integrator_urdf += "/test_data/integrator.urdf";

    // No contact_info
    RobotContactInfo contact_info;

    FullOrderRigidBody int_model(pin_model_name, integrator_urdf);

    REQUIRE(int_model.GetMass() == 1);
}

TEST_CASE("Hopper", "[model][pinocchio]") {
    using namespace torc::models;
    const std::string pin_model_name = "test_pin_model";

    std::filesystem::path hopper_urdf = std::filesystem::current_path();
    hopper_urdf += "/test_data/hopper.urdf";

    RobotContactInfo contact_info;
    contact_info.contacts.emplace("foot", Contact(PointContact, true));

    FullOrderRigidBody hopper_model(pin_model_name, hopper_urdf);

    int constexpr INPUT_SIZE = 4;
    constexpr int CONFIG_SIZE = 11;
    constexpr int VEL_SIZE = 10;
    constexpr int STATE_SIZE = 21;
    constexpr int DERIV_SIZE = 20;
    constexpr int JOINT_SIZE = 6;

    vectorx_t x = FullOrderRigidBody::BuildState(hopper_model.GetRandomConfig(), hopper_model.GetRandomVel());
    const vectorx_t input = vectorx_t::Zero(INPUT_SIZE);
    vectorx_t xdot = hopper_model.GetDynamics(x, input, contact_info);

    REQUIRE(hopper_model.GetNumInputs() == INPUT_SIZE);
    REQUIRE(hopper_model.GetConfigDim() == CONFIG_SIZE);
    REQUIRE(hopper_model.GetVelDim() == VEL_SIZE);
    REQUIRE(hopper_model.GetStateDim() == STATE_SIZE);
    REQUIRE(hopper_model.GetDerivativeDim() == DERIV_SIZE);
    REQUIRE(hopper_model.GetNumJoints() == JOINT_SIZE);
    REQUIRE(hopper_model.GetSystemType() == HybridSystemImpulse);

    SECTION("Dynamics Derivatives") {
        CheckDerivatives(hopper_model);
    }

    SECTION("Contact Dynamics Derivatives") {
        CheckContactDerivatives(hopper_model, contact_info);

        contact_info.contacts.at("foot").state = false;
        CheckImpulseDerivatives(hopper_model, contact_info);
    }

    SECTION("Impulse Dynamics Derivatives") {
        CheckImpulseDerivatives(hopper_model, contact_info);

        contact_info.contacts.at("foot").state = false;
        CheckImpulseDerivatives(hopper_model, contact_info);
    }
}

class ModelTester : public torc::models::FullOrderRigidBody {
public:
    ModelTester(const std::string& name, const std::filesystem::path& path) : torc::models::FullOrderRigidBody(name, path) {}

    torc::models::vectorx_t GetDynamicsFullTau(const torc::models::vectorx_t& q, const torc::models::vectorx_t& v, const torc::models::vectorx_t& tau) {
        pinocchio::aba(pin_model_, *pin_data_, q, v, tau);
        return this->BuildStateDerivative(v, pin_data_->ddq);
    }
};

//TEST_CASE("Quadruped Inverse Dynamics", "[model][pinocchio][id]") {
//    using namespace torc::models;
//    const std::string pin_model_name = "test_pin_model";
//
//    constexpr double MARGIN = 1e-6;
//
//    std::filesystem::path a1_urdf = std::filesystem::current_path();
//    a1_urdf += "/test_data/test_a1.urdf";
//
//    ModelTester a1_model(pin_model_name, a1_urdf);
//
//    constexpr int INPUT_SIZE = 12;
//    constexpr int STATE_SIZE = 37;
//
//    // ----------------------------------------------- //
//    // --------------- Inverse Dynamics -------------- //
//    // ----------------------------------------------- //
//    for (int i = 0; i < 10; i++) {
//        vectorx_t input = vectorx_t::Zero(INPUT_SIZE);
//        input.setRandom();
//        vectorx_t x = a1_model.GetRandomState();
//        vectorx_t a(a1_model.GetVelDim());
//        a.setRandom();
//        a = 10 * a;
//
//        // Get some random forces
//        std::vector<ExternalForce> forces;
//        forces.emplace_back("FR_foot", vector3_t::Zero());
//        forces.emplace_back("FL_foot", vector3_t::Zero());
//
//        // Get the inverse dynamics
//        vectorx_t tau = a1_model.InverseDynamics(x, a, forces);
//
//        // With zero external force, the result from the inverse dynamics should match the result from the forward dynamics
//        vectorx_t q, v;
//        a1_model.ParseState(x, q, v);
//        vectorx_t xdot = a1_model.GetDynamicsFullTau(q, v, tau);
//
//        REQUIRE(VectorEqualWithMargin(xdot.tail(18), a, MARGIN));
//    }
//}

TEST_CASE("Quadruped Inverse Dynamics Derivatives", "[model][pinocchio][id][derivatves]") {
    using namespace torc::models;
    const std::string pin_model_name = "test_pin_model";

    constexpr double MARGIN = 1e-6;

    std::filesystem::path a1_urdf = std::filesystem::current_path();
    a1_urdf += "/test_data/test_a1.urdf";

    FullOrderRigidBody a1_model(pin_model_name, a1_urdf);

    constexpr int INPUT_SIZE = 12;
    constexpr int STATE_SIZE = 37;

    // ----------------------------------------------- //
    // --------- Inverse Dynamics Derivatives -------- //
    // ----------------------------------------------- //
    std::vector<std::string> contact_names;
    contact_names.emplace_back("FR_foot_fixed");
    contact_names.emplace_back("FL_foot_fixed");
    contact_names.emplace_back("RR_foot");
    contact_names.emplace_back("RL_foot");

    CheckInverseDynamicsDerivatives(a1_model, contact_names);
//    for (int i = 0; i < 1; i++) {
//        vectorx_t input = vectorx_t::Zero(INPUT_SIZE);
//        input.setRandom();
//        vectorx_t x = a1_model.GetRandomState();
//        vectorx_t a(a1_model.GetVelDim());
//        a.setRandom();
//        a = 10 * a;
//
//        // Get some random forces
//        std::vector<ExternalForce> forces;
//        forces.emplace_back("FR_foot", vector3_t::Zero());
//        forces.emplace_back("FL_foot", vector3_t::Zero());
//
//        // Get the inverse dynamics
//        matrixx_t dtau_dq, dtau_dv, dtau_da;
//        dtau_dq.setZero(a1_model.GetVelDim(), a1_model.GetVelDim());
//        dtau_dv.setZero(a1_model.GetVelDim(), a1_model.GetVelDim());
//        dtau_da.setZero(a1_model.GetVelDim(), a1_model.GetVelDim());
//
//        a1_model.InverseDynamicsDerivative(x, a, forces, dtau_dq, dtau_dv, dtau_da);
//
////        REQUIRE(VectorEqualWithMargin(xdot.tail(18), a, MARGIN));
//    }
}
