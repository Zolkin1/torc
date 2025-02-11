//
// Created by zolkin on 1/18/25.
//

#include <eigen_utils.h>
#include <torc_timer.h>
#include <catch2/catch_test_macros.hpp>

#include "constraints/DynamicsConstraintsTest.h"
#include "contact_schedule.h"
#include "pinocchio_interface.h"
#include "simple_trajectory.h"
#include "MpcSettings.h"

#define ENABLE_BENCHMARKS false

// TEST_CASE("Constraints Test", "[mpc]") {
//     using namespace torc::mpc;
//     const std::string pin_model_name = "test_pin_model";
//     std::filesystem::path a1_urdf = std::filesystem::current_path();
//     a1_urdf += "/test_data/test_a1.urdf";
//
//     std::filesystem::path mpc_config = std::filesystem::current_path();
//     mpc_config += "/test_data/mpc_config.yaml";
//
//     torc::models::FullOrderRigidBody a1("a1", a1_urdf);
//
//     fs::path deriv_lib_path = fs::current_path();
//     deriv_lib_path = deriv_lib_path / "deriv_libs";
//
//     std::vector<std::string> contact_frames = {"RL_foot", "FR_foot", "FL_foot", "RR_foot"};
//     DynamicsConstraint dynamics(a1, contact_frames, "a1_full_order", deriv_lib_path, true, true, 0, 5);
// }

TEST_CASE("Forward Dynamics") {
    using namespace torc::mpc;
    const std::string pin_model_name = "test_pin_model";
    std::filesystem::path a1_urdf = std::filesystem::current_path();
    a1_urdf += "/test_data/test_a1.urdf";

    std::filesystem::path mpc_config = std::filesystem::current_path();
    mpc_config += "/test_data/mpc_config.yaml";

    std::vector<std::string> contact_frames = {"RL_foot", "FR_foot", "FL_foot", "RR_foot"};
    torc::models::FullOrderRigidBody a1("a1", a1_urdf);

    for (int i = 0; i < 10; i++) {
        // Get random config
        vectorx_t q = a1.GetRandomConfig();

        // Get random vel
        vectorx_t v = a1.GetRandomVel();

        //Get random torques
        vectorx_t tau = a1.GetRandomVel().tail(v.size());

        // Get random forces
        std::vector<torc::models::ExternalForce<double>> f_ext;
        for (const auto& f : contact_frames) {
            torc::models::ExternalForce<double> force(f, vector3_t::Random());  // TODO: make sure random isn't crazy
            f_ext.emplace_back(force);
        }

        // Call ForwardDynamics
        pinocchio::Data data(a1.GetModel());
        vectorx_t a = torc::models::ForwardDynamics<double>(a1.GetModel(), data,
            q, v, tau, f_ext);

        // Call ForwardWithCrba
        vectorx_t a_crba = torc::models::ForwardWithCrba<double>(a1.GetModel(), data,
            q, v, tau, f_ext);

        // Check
        CHECK(a.isApprox(a_crba));
    }
}

// TODO: Come back to this
TEST_CASE("Dynamics Constraint") {
    using namespace torc::mpc;
    std::filesystem::path g1_urdf = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/g1_hand.urdf";

    std::filesystem::path mpc_config = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/g1_mpc_config.yaml";

    MpcSettings settings(mpc_config);
    settings.Print();

    torc::models::FullOrderRigidBody g1("g1", g1_urdf, settings.joint_skip_names, settings.joint_skip_values);
    std::cout << "Model Created." << std::endl;

    auto curr_path = fs::current_path();
    curr_path = curr_path / "dynamics_constraint_deriv_libs";

    // DynamicsConstraintsTest c1(g1, contact_frames, "g1_c1", curr_path, false, false, //true, true,
    //     0, 5);

    // TODO: Pick different configs
    vectorx_t q1 = g1.GetRandomConfig();
    vectorx_t q2 = g1.GetRandomConfig();

    vectorx_t v1 = g1.GetRandomVel();
    vectorx_t v2 = g1.GetRandomVel();

    vectorx_t tau = vectorx_t::Random(g1.GetVelDim() - 6);
    vectorx_t tau_pin = vectorx_t::Zero(g1.GetVelDim());
    tau_pin << vectorx_t::Zero(6), tau;

    std::vector<torc::models::ExternalForce<double>> f_ext;
    for (const auto& f : settings.contact_frames) {
        f_ext.emplace_back(f, vector3_t::Random());
    }

    pinocchio::Data data(g1.GetModel());
    vectorx_t a_fd = torc::models::ForwardDynamics<double>(g1.GetModel(), data, q1, v1, tau_pin, f_ext);
    vectorx_t tau_id = torc::models::InverseDynamics(g1.GetModel(), data, q1, v1, a_fd, f_ext);

    std::cout << "tau: " << tau_pin.transpose() << std::endl;
    std::cout << "tau_id: " << tau_id.transpose() << std::endl;

    CHECK(tau_pin.isApprox(tau_id));

    vectorx_t a_approx = (v2 - v1)/settings.dt[0];
    tau_id = torc::models::InverseDynamics(g1.GetModel(), data, q1, v1, a_approx, f_ext);
    a_fd = torc::models::ForwardDynamics(g1.GetModel(), data, q1, v1, tau_id, f_ext);

    std::cout << "a approx: " << a_approx.transpose() << std::endl;
    std::cout << "a_fd: " << a_fd.transpose() << std::endl;
    CHECK(a_fd.isApprox(a_approx));
}

