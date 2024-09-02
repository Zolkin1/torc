//
// Created by zolkin on 6/20/24.
//

#include <eigen_utils.h>
#include <torc_timer.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "full_order_mpc.h"
#include "contact_schedule.h"

#define ENABLE_BENCHMARKS false

TEST_CASE("A1 MPC Test", "[mpc]") {
    using namespace torc::mpc;
    const std::string pin_model_name = "test_pin_model";
    std::filesystem::path a1_urdf = std::filesystem::current_path();
    a1_urdf += "/test_data/test_a1.urdf";

    std::filesystem::path mpc_config = std::filesystem::current_path();
    mpc_config += "/test_data/mpc_config.yaml";

    FullOrderMpc mpc("a1_mpc", mpc_config, a1_urdf);

    mpc.Configure();

    torc::models::FullOrderRigidBody a1("a1", a1_urdf);

    vectorx_t random_state(a1.GetConfigDim() + a1.GetVelDim());
    vectorx_t q_rand = a1.GetRandomConfig();
    vectorx_t v_rand = a1.GetRandomVel();
    // random_state.tail(a1.GetVelDim()).setZero();
    std::cout << "initial config: " << q_rand.transpose() << std::endl;
    std::cout << "initial vel: " << v_rand.transpose() << std::endl;
    Trajectory traj;
    traj.UpdateSizes(a1.GetConfigDim(), a1.GetVelDim(), a1.GetNumInputs(), mpc.GetContactFrames(), mpc.GetNumNodes());

    vectorx_t q_neutral = a1.GetNeutralConfig();
    q_neutral(2) += 0.321;
    traj.SetDefault(q_neutral);

    mpc.SetWarmStartTrajectory(traj);
    mpc.SetConstantConfigTarget(q_neutral);
    mpc.SetConstantVelTarget(vectorx_t::Zero(a1.GetVelDim()));

    ContactSchedule cs(mpc.GetContactFrames());
    const double swing_time = 0.3;
    double time = 0;
    for (int i = 0; i < 3; i++) {
        if (i % 2 != 0) {
            cs.InsertSwing("FR_foot", time, time + swing_time);
            cs.InsertSwing("RL_foot", time, time + swing_time);
        } else {
            cs.InsertSwing("FL_foot", time, time + swing_time);
            cs.InsertSwing("RR_foot", time, time + swing_time);
        }
        time += swing_time;
    }

    mpc.UpdateContactSchedule(cs);

    mpc.ComputeNLP(q_rand, v_rand, traj);

    for (int i = 0; i < traj.GetNumNodes(); i++) {
        std::cout << "Node: " << i << std::endl;
        std::cout << "config: " << traj.GetConfiguration(i).transpose() << std::endl;
        std::cout << "vel: " << traj.GetVelocity(i).transpose() << std::endl;
        std::cout << "torque: " << traj.GetTau(i).transpose() << std::endl;
    }

    // random_state = a1.GetRandomState();
    // mpc.Compute(q_rand, v_rand, traj);

    mpc.PrintStatistics();
    std::cout << std::endl << std::endl;
    mpc.PrintContactSchedule();
    std::cout << std::endl << std::endl;
}

TEST_CASE("Achilles MPC Test", "[mpc]") {
    using namespace torc::mpc;
    const std::string pin_model_name = "test_pin_model";
    std::filesystem::path achilles_urdf = std::filesystem::current_path();
    achilles_urdf += "/test_data/achilles.urdf";

    std::filesystem::path mpc_config = std::filesystem::current_path();
    mpc_config += "/test_data/achilles_mpc_config.yaml";

    FullOrderMpc mpc("achilles_mpc", mpc_config, achilles_urdf);

    mpc.Configure();

    torc::models::FullOrderRigidBody achilles("achilles", achilles_urdf);

    vectorx_t random_state(achilles.GetConfigDim() + achilles.GetVelDim());
    vectorx_t q_rand = achilles.GetRandomConfig();
    vectorx_t v_rand = achilles.GetRandomVel();

    // random_state.tail(a1.GetVelDim()).setZero();
    std::cout << "initial config: " << random_state.head(achilles.GetConfigDim()).transpose() << std::endl;
    std::cout << "initial vel: " << random_state.tail(achilles.GetVelDim()).transpose() << std::endl;

    ContactSchedule cs(mpc.GetContactFrames());
    const double swing_time = 0.3;
    double time = 0;
    for (int i = 0; i < 3; i++) {
        if (i % 2 != 0) {
            cs.InsertSwing("foot_front_right", time, time + swing_time);
            cs.InsertSwing("foot_rear_right", time, time + swing_time);
            // cs.InsertContact("right_hand", time, time + contact_time);
        } else {
            cs.InsertSwing("foot_front_left", time, time + swing_time);
            cs.InsertSwing("foot_rear_left", time, time + swing_time);
            // cs.InsertContact("left_hand", time, time + contact_time);
        }
        time += swing_time;
    }

    mpc.UpdateContactSchedule(cs);

    vectorx_t q_target;
    q_target.resize(achilles.GetConfigDim());
    q_target << 0, 0, 0.97,
                0, 0, 0, 1,
                0, 0, -0.26,
                0, 0.65, -0.43,
                0, 0, 0,
                0, 0, -0.26,
                0.65, -0.43,
                0, 0, 0;
    mpc.SetConstantConfigTarget(q_target);

    Trajectory traj;
    traj.UpdateSizes(achilles.GetConfigDim(), achilles.GetVelDim(), achilles.GetNumInputs(), mpc.GetContactFrames(), mpc.GetNumNodes());

    vectorx_t q_neutral = achilles.GetNeutralConfig();
    q_neutral(2) += 0.321;
    traj.SetDefault(q_neutral);

    mpc.SetWarmStartTrajectory(traj);

    mpc.SetConstantConfigTarget(q_neutral);
    mpc.SetConstantVelTarget(vectorx_t::Zero(achilles.GetVelDim()));

    mpc.Compute(q_rand, v_rand, traj);

    // for (int i = 0; i < traj.GetNumNodes(); i++) {
    //     std::cout << "Node: " << i << std::endl;
    //     std::cout << "config: " << traj.GetConfiguration(i).transpose() << std::endl;
    //     std::cout << "vel: " << traj.GetVelocity(i).transpose() << std::endl;
    //     std::cout << "torque: " << traj.GetTau(i).transpose() << std::endl;
    // }

    mpc.Compute(q_rand, v_rand, traj);

    mpc.PrintStatistics();
    std::cout << std::endl << std::endl;
    mpc.PrintContactSchedule();
    std::cout << std::endl << std::endl;
}

TEST_CASE("Contact schedule", "[mpc][contact schedule]") {
    std::cout << "Contact Schedule Tests" << std::endl;
    torc::mpc::ContactSchedule cs({"LF_FRONT", "RF_FRONT"});
    cs.InsertSwing("LF_FRONT", 0.1, 0.2);
    cs.InsertSwing("RF_FRONT", 0.05, 0.15);

    CHECK(cs.InContact("LF_FRONT", 0.075));
    CHECK(cs.InContact("LF_FRONT", 0.25));

    CHECK(cs.InSwing("LF_FRONT", 0.12));
    CHECK(cs.InSwing("LF_FRONT", 0.2));

    CHECK(cs.InSwing("RF_FRONT", 0.1));
    CHECK(cs.InSwing("RF_FRONT", 0.12));
    CHECK(cs.InSwing("RF_FRONT", 0.05));
    CHECK(cs.InContact("RF_FRONT", 0.025));
    CHECK(cs.InContact("RF_FRONT", 0.25));
}

TEST_CASE("Trajectory Interpolation", "[trajectory]") {
    std::cout << "====== Trajectory Interpolation =====" << std::endl;
    using namespace torc::mpc;

    std::filesystem::path achilles_urdf = std::filesystem::current_path();
    achilles_urdf += "/test_data/achilles.urdf";
    torc::models::FullOrderRigidBody achilles("achilles", achilles_urdf);

    std::filesystem::path mpc_config = std::filesystem::current_path();
    mpc_config += "/test_data/achilles_mpc_config.yaml";

    FullOrderMpc mpc("achilles_mpc", mpc_config, achilles_urdf);

    Trajectory traj;
    traj.UpdateSizes(achilles.GetConfigDim(), achilles.GetVelDim(), achilles.GetNumInputs(),
        mpc.GetContactFrames(), mpc.GetNumNodes());

    vectorx_t q_neutral = achilles.GetNeutralConfig();
    traj.SetDefault(q_neutral);

    std::vector<double> dt(mpc.GetNumNodes() - 1);
    std::fill(dt.begin(), dt.end(), 0.02);
    traj.SetDtVector(dt);

    // ----- Configuration ----- //
    // Checks when all the configurations are the same
    vectorx_t q_interp;
    traj.GetConfigInterp(0, q_interp);
    CHECK(q_interp.isApprox(q_neutral));

    traj.GetConfigInterp(0.1, q_interp);
    CHECK(q_interp.isApprox(q_neutral));

    traj.GetConfigInterp(0.213, q_interp);
    CHECK(q_interp.isApprox(q_neutral));

    // Checks when there are differences
    vectorx_t q_rand = achilles.GetRandomConfig();
    traj.SetConfiguration(1, q_rand);

    traj.GetConfigInterp(0, q_interp);
    CHECK(q_interp.isApprox(q_neutral));

    traj.GetConfigInterp(0.01, q_interp);
    // This is the expected vector EXCEPT for the quaternion part
    vectorx_t q_expect = 0.5*q_rand + 0.5*q_neutral;

    CHECK(q_interp.head<3>().isApprox(q_expect.head<3>()));
    CHECK(q_interp.tail(achilles.GetNumInputs()).isApprox(q_expect.tail(achilles.GetNumInputs())));
    // TODO: Add check for quaternion values

    // ----- Velocity ----- //
    // Checks when all the configurations are the same
    vectorx_t v_zero = vectorx_t::Zero(achilles.GetVelDim());
    vectorx_t v_interp;
    traj.GetVelocityInterp(0, v_interp);
    CHECK(v_interp.isApprox(v_zero));

    traj.GetVelocityInterp(0.1, v_interp);
    CHECK(v_interp.isApprox(v_zero));

    traj.GetVelocityInterp(0.213, v_interp);
    CHECK(v_interp.isApprox(v_zero));

    // Checks when there are differences
    vectorx_t v_rand = achilles.GetRandomVel();
    traj.SetVelocity(1, v_rand);

    traj.GetVelocityInterp(0, v_interp);
    CHECK(v_interp.isApprox(v_zero));

    traj.GetVelocityInterp(0.01, v_interp);
    vectorx_t v_expect = 0.5*v_rand + 0.5*v_zero;
    CHECK(v_interp.isApprox(v_expect));

    traj.GetVelocityInterp(0.005, v_interp);
    v_expect = 0.25*v_rand + 0.75*v_zero;
    CHECK(v_interp.isApprox(v_expect));

    // ---- Torque ----- //
    // (Mostly covered by velocity because they use the same internals)
    vectorx_t tau_interp;
    vectorx_t tau_zero = vectorx_t::Zero(achilles.GetNumInputs());

    vectorx_t tau_rand = vectorx_t::Random(achilles.GetNumInputs());
    traj.SetTau(1, tau_rand);

    traj.GetTorqueInterp(0.01, tau_interp);
    vectorx_t tau_expect = 0.5*tau_rand + 0.5*tau_zero;
    CHECK(tau_interp.isApprox(tau_expect));

    // TODO: Add force checks
}

#if ENABLE_BENCHMARKS
TEST_CASE("MPC Benchmarks [A1]", "[mpc][benchmarks]") {
    // Benchmarking with the A1
    using namespace torc::mpc;
    const std::string pin_model_name = "test_pin_model";
    std::filesystem::path a1_urdf = std::filesystem::current_path();
    a1_urdf += "/test_data/test_a1.urdf";

    std::filesystem::path mpc_config = std::filesystem::current_path();
    mpc_config += "/test_data/mpc_config.yaml";

    FullOrderMpc mpc(mpc_config, a1_urdf);
    mpc.SetVerbosity(false);

    BENCHMARK("MPC Configure [A1]") {
        mpc.Configure();
    };

    torc::models::FullOrderRigidBody a1("a1", a1_urdf);
    vectorx_t random_state = a1.GetRandomState();
    BENCHMARK("MPC Compute [A1]") {
        mpc.Compute(random_state);
    };
}
#endif
