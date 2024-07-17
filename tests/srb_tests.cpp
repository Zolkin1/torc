//
// Created by zolkin on 6/6/24.
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "full_order_rigid_body.h"
#include "single_rigid_body.h"

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

TEST_CASE("SRB Quadruped", "[model][pinocchio][srb][benchmarks]") {
    using namespace torc::models;

    std::filesystem::path a1_urdf = std::filesystem::current_path();
    a1_urdf += "/test_data/test_a1.urdf";

    constexpr int MAX_CONTACTS = 4;

    SingleRigidBody a1_model("a1", a1_urdf, MAX_CONTACTS);

    constexpr int INPUT_SIZE = MAX_CONTACTS*6;
    constexpr int CONFIG_SIZE = SingleRigidBody::SRB_CONFIG_DIM;
    constexpr int VEL_SIZE = SingleRigidBody::SRB_VEL_DIM;
    constexpr int STATE_SIZE = CONFIG_SIZE + VEL_SIZE;
    constexpr int DERIV_SIZE = VEL_SIZE*2;
    constexpr int JOINT_SIZE = 2;

    REQUIRE(a1_model.GetNumInputs() == INPUT_SIZE);
    REQUIRE(a1_model.GetConfigDim() == CONFIG_SIZE);
    REQUIRE(a1_model.GetVelDim() == VEL_SIZE);
    REQUIRE(a1_model.GetStateDim() == STATE_SIZE);
    REQUIRE(a1_model.GetDerivativeDim() == DERIV_SIZE);
    REQUIRE(a1_model.GetNumJoints() == JOINT_SIZE);
    REQUIRE(a1_model.GetSystemType() == HybridSystemNoImpulse);

    SECTION("Updated Reference Config") {
        FullOrderRigidBody full_a1_model("a1_full", a1_urdf);
        vectorx_t x_rand = full_a1_model.GetRandomState();
        vectorx_t q, v;
        full_a1_model.ParseState(x_rand, q, v);
        a1_model.SetRefConfig(q);

        REQUIRE(a1_model.GetNumInputs() == INPUT_SIZE);
        REQUIRE(a1_model.GetConfigDim() == CONFIG_SIZE);
        REQUIRE(a1_model.GetVelDim() == VEL_SIZE);
        REQUIRE(a1_model.GetStateDim() == STATE_SIZE);
        REQUIRE(a1_model.GetDerivativeDim() == DERIV_SIZE);
        REQUIRE(a1_model.GetNumJoints() == JOINT_SIZE);
        REQUIRE(a1_model.GetSystemType() == HybridSystemNoImpulse);

        REQUIRE(VectorEqualWithMargin(q, a1_model.GetRefConfig(), 1e-8));
    }

    matrixx_t A, B;
    A = matrixx_t::Zero(a1_model.GetDerivativeDim(), a1_model.GetDerivativeDim());
    B = matrixx_t::Zero(a1_model.GetDerivativeDim(), a1_model.GetNumInputs());

    vectorx_t x_rand = a1_model.GetRandomState();
    vectorx_t input_rand;
    input_rand.setRandom(a1_model.GetNumInputs());

    // Benchmarks
    BENCHMARK("Dynamics Derivatives") {
        return a1_model.DynamicsDerivative(x_rand, input_rand, A, B);
    };

    BENCHMARK("Dynamics") {
        return a1_model.GetDynamics(x_rand, input_rand);
    };
}
