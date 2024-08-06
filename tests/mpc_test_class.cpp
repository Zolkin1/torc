//
// Created by zolkin on 7/31/24.
//


#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "mpc_test_class.h"
#include "cost_test_class.h"
#include "autodiff_fn.h"

TEST_CASE("MPC Test Class", "[mpc]") {
    using namespace torc::mpc;
    const std::string pin_model_name = "test_pin_model";
    std::filesystem::path a1_urdf = std::filesystem::current_path();
    a1_urdf += "/test_data/test_a1.urdf";

    std::filesystem::path mpc_config = std::filesystem::current_path();
    mpc_config += "/test_data/mpc_config.yaml";

    MpcTestClass mpc(mpc_config, a1_urdf);

    mpc.Configure();

    mpc.CheckQuaternionIntLin();
    mpc.CheckInverseDynamicsLin();
    mpc.CheckQuaternionLin();
    mpc.CheckSwingHeightLin();
    mpc.CheckHolonomicLin();
    mpc.CheckCostFunctionDefiniteness();
    mpc.CheckConstraintIdx();
    // mpc.BenchmarkQuaternionIntegrationLin();
    // mpc.BenchmarkInverseDynamicsLin();
    // mpc.BenchmarkQuaternionConfigurationLin();
    // mpc.BenchmarkSwingHeightLin();
    // mpc.BenchmarkHolonomicLin();
    // mpc.BenchmarkConstraints();
    // mpc.BenchmarkCompute();
    // mpc.BenchmarkCostFunctions();
}

TEST_CASE("Cost Test Class", "[mpc][cost]") {
    using namespace torc::mpc;
    const std::string pin_model_name = "test_pin_model";
    std::filesystem::path a1_urdf = std::filesystem::current_path();
    a1_urdf += "/test_data/test_a1.urdf";

    // std::filesystem::path mpc_config = std::filesystem::current_path();
    // mpc_config += "/test_data/mpc_config.yaml";

    torc::models::FullOrderRigidBody a1_model("a1_test_model", a1_urdf);

    CostTestClass cost_fcn(a1_model);

    cost_fcn.CheckConfigure();
    cost_fcn.CheckDerivatives();
    cost_fcn.CheckDefaultCosts();
    cost_fcn.CheckLinearizeQuadrasize();

    // cost_fcn.BenchmarkCostFunctions();
}
