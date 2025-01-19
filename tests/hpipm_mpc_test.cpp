//
// Created by zolkin on 1/18/25.
//

#include <eigen_utils.h>
#include <torc_timer.h>
#include <catch2/catch_test_macros.hpp>

#include "DynamicsConstraint.h"
#include "contact_schedule.h"
#include "pinocchio_interface.h"
#include "simple_trajectory.h"

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