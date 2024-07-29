//
// Created by zolkin on 6/20/24.
//

#include <catch2/catch_test_macros.hpp>

#include "full_order_mpc.h"

TEST_CASE("Basic MPC Test", "[mpc]") {
    using namespace torc::mpc;
    const std::string pin_model_name = "test_pin_model";
    std::filesystem::path a1_urdf = std::filesystem::current_path();
    a1_urdf += "/test_data/test_a1.urdf";

    std::filesystem::path mpc_config = std::filesystem::current_path();
    mpc_config += "/test_data/mpc_config.yaml";

    FullOrderMpc mpc(mpc_config, a1_urdf);

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