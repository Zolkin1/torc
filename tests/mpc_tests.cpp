//
// Created by zolkin on 6/20/24.
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "full_order_mpc.h"

TEST_CASE("Basic MPC Test", "[mpc]") {
    using namespace torc::mpc;
    const std::string pin_model_name = "test_pin_model";
    std::filesystem::path a1_urdf = std::filesystem::current_path();
    a1_urdf += "/test_data/test_a1.urdf";

    std::filesystem::path mpc_config = std::filesystem::current_path();
    mpc_config += "/test_data/mpc_config.yaml";

    FullOrderMpc mpc(mpc_config, a1_urdf);

    mpc.Configure();


    torc::models::FullOrderRigidBody a1("a1", a1_urdf);

    vectorx_t random_state = a1.GetRandomState();
    mpc.Compute(random_state);

    random_state = a1.GetRandomState();
    mpc.Compute(random_state);

//    torc::models::RigidBody a1_model(pin_model_name, a1_urdf);
//
//    constexpr int NODES = 10;
//    MPCContact mpc(a1_model, NODES);
//
//    ContactTrajectory traj;
//    traj.Reset(NODES, a1_model.GetConfigDim(), a1_model.GetVelDim());
//
//    mpc.ToBilateralData(traj);
}

//TEST_CASE("MPC Benchmarks [A1]", "[mpc][benchmarks]") {
//    // Benchmarking with the A1
//    using namespace torc::mpc;
//    const std::string pin_model_name = "test_pin_model";
//    std::filesystem::path a1_urdf = std::filesystem::current_path();
//    a1_urdf += "/test_data/test_a1.urdf";
//
//    std::filesystem::path mpc_config = std::filesystem::current_path();
//    mpc_config += "/test_data/mpc_config.yaml";
//
//    FullOrderMpc mpc(mpc_config, a1_urdf);
//    mpc.SetVerbosity(false);
//
//    BENCHMARK("MPC Configure [A1]") {
//        mpc.Configure();
//    };
//
//    torc::models::FullOrderRigidBody a1("a1", a1_urdf);
//    vectorx_t random_state = a1.GetRandomState();
//    BENCHMARK("MPC Compute [A1]") {
//        mpc.Compute(random_state);
//    };
//}