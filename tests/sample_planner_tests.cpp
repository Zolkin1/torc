//
// Created by zolkin on 8/7/24.
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include <filesystem>

#include "full_order_rigid_body.h"
#include "cross_entropy.h"
#include "simulation_dispatcher_test_class.h"

TEST_CASE("Basic Sample Planner Test", "[sample_planner]") {
    using namespace torc::sample;

    // XML
    std::filesystem::path achilles_xml = std::filesystem::current_path();
    achilles_xml += "/test_data/achilles.xml";

    // Config yaml
    std::filesystem::path cem_config = std::filesystem::current_path();
    cem_config += "/test_data/cem_config.yaml";

    // Robot Model
    std::filesystem::path achilles_urdf = std::filesystem::current_path();
    achilles_urdf += "/test_data/achilles.urdf";

    torc::models::FullOrderRigidBody achilles("achilles", achilles_urdf);

    // MPC
    std::filesystem::path mpc_config = std::filesystem::current_path();
    mpc_config += "/test_data/achilles_mpc_config_sim.yaml";

    std::shared_ptr<torc::mpc::FullOrderMpc> mpc;
    mpc = std::make_shared<torc::mpc::FullOrderMpc>("achilles_test_class_cem", mpc_config, achilles_urdf);

    mpc->Configure();

    vectorx_t q_target, v_target;
    q_target.resize(achilles.GetConfigDim());
    q_target << 0., 0, 0.97,    // position
            0, 0, 0, 1,     // quaternion
            0, 0, -0.26,    // L hips joints
            0.65, -0.43,    // L knee, ankle
            0, 0, 0, 0,     // L shoulders and elbow
            0, 0, -0.26,    // R hip joints
            0.65, -0.43,    // R knee ankle
            0, 0, 0, 0;     // R shoulders and elbow

    mpc->SetConstantConfigTarget(q_target);
    q_target(0) = 0;

    v_target.resize(achilles.GetVelDim());
    v_target << 0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0;
    mpc->SetConstantVelTarget(v_target);

    unsigned int num_threads = std::thread::hardware_concurrency()-2;
    std::cout << "Number of threads being used: " << num_threads << std::endl;

    // CEM
    CrossEntropy cem(achilles_xml, 100, cem_config, mpc, num_threads);

    // Reference trajectory
    torc::mpc::Trajectory traj_ref;
    const int NUM_NODES = 32;
    traj_ref.UpdateSizes(achilles.GetConfigDim(), achilles.GetVelDim(), achilles.GetNumInputs(), {"right_foot", "left_foot"}, NUM_NODES);
    traj_ref.SetDefault(q_target);
    std::vector<double> dt_vec(NUM_NODES);
    std::fill(dt_vec.begin(), dt_vec.end(), 0.025);

    torc::mpc::Trajectory traj_out;

    traj_ref.SetDtVector(dt_vec);

    std::cout << "Starting variance: " << cem.GetVariance() << std::endl;
    std::cout << "Starting cost avg: " << cem.GetAvgCost() << std::endl;

    torc::mpc::ContactSchedule cs_out;
    cem.Plan(traj_ref, traj_out, cs_out, {"ConfigTracking", "VelocityTracking"});

    std::cout << "Config node 0: " << traj_out.GetConfiguration(0).transpose() << std::endl;
    std::cout << "Config node 1: " << traj_out.GetConfiguration(1).transpose() << std::endl;

    std::cout << "Variance after solve 1: " << cem.GetVariance() << std::endl;
    std::cout << "Cost avg after solve 1: " << cem.GetAvgCost() << std::endl;

    for (int i = 0; i < 300; i++) {
        cem.Plan(traj_ref, traj_out, cs_out, {"ConfigTracking", "VelocityTracking"});
        traj_ref = traj_out;
    }

    std::cout << "Config node 0: " << traj_out.GetConfiguration(0).transpose() << std::endl;
    std::cout << "Config node 1: " << traj_out.GetConfiguration(1).transpose() << std::endl;

//    BENCHMARK("cem plan") {
//        cem.Plan(traj_ref, traj_out, cs_out, {"ConfigTracking", "VelocityTracking"});
//    };

    std::cout << "Ending variance: " << cem.GetVariance() << std::endl;
    std::cout << "Ending cost avg: " << cem.GetAvgCost() << std::endl;
}

TEST_CASE("Simulation Dispatcher Class Test", "[sample_planner]") {
    using namespace torc::sample;

    std::filesystem::path achilles_xml = std::filesystem::current_path();
    achilles_xml += "/test_data/achilles.xml";

    std::filesystem::path achilles_urdf = std::filesystem::current_path();
    achilles_urdf += "/test_data/achilles.urdf";
    torc::models::FullOrderRigidBody achilles("achilles", achilles_urdf);

    SimulationDispatcherTest simulation_dispatcher(achilles_xml, 60);
    simulation_dispatcher.CheckModelName("achilles");
    simulation_dispatcher.CheckActuatedJoints();

    torc::mpc::Trajectory traj_ref;
    const int NUM_NODES = 30;
    traj_ref.UpdateSizes(achilles.GetConfigDim(), achilles.GetVelDim(), achilles.GetNumInputs(), {"right_foot", "left_foot"}, NUM_NODES);
    traj_ref.SetDefault(achilles.GetRandomConfig());
    std::vector<double> dt_vec(NUM_NODES);
    std::fill(dt_vec.begin(), dt_vec.end(), 0.02);

    traj_ref.SetDtVector(dt_vec);

    simulation_dispatcher.CheckSingleSimuation(traj_ref);
//    simulation_dispatcher.CheckBatchSimulation(traj_ref);

    simulation_dispatcher.CheckQuaternionConversions();

    // simulation_dispatcher.BenchmarkSims(traj_ref);
}