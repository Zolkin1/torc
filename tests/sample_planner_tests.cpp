//
// Created by zolkin on 8/7/24.
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include <filesystem>
#include "absl/log/initialize.h"

#include "full_order_rigid_body.h"
#include "cross_entropy.h"
#include "simulation_dispatcher_test_class.h"

TEST_CASE("Basic Sample Planner Test", "[sample_planner]") {
    using namespace torc::sample;

    std::filesystem::path achilles_xml = std::filesystem::current_path();
    achilles_xml += "/test_data/achilles.xml";

    CrossEntropy cem(achilles_xml, 10);
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
    simulation_dispatcher.CheckBatchSimulation(traj_ref);

    simulation_dispatcher.BenchmarkSims(traj_ref);
}