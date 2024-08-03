//
// Created by zolkin on 7/31/24.
//


#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "mpc_test_class.h"

#include "autodiff_fn.h"

TEST_CASE("MPC Test Class", "[mpc]") {
    using namespace torc::mpc;
    const std::string pin_model_name = "test_pin_model";
    std::filesystem::path a1_urdf = std::filesystem::current_path();
    a1_urdf += "/test_data/test_a1.urdf";

    std::filesystem::path mpc_config = std::filesystem::current_path();
    mpc_config += "/test_data/mpc_config.yaml";

    MpcTestClass mpc(mpc_config, a1_urdf);

    mpc.CheckQuaternionIntLin();
    mpc.CheckInverseDynamicsLin();
    mpc.CheckQuaternionLin();
    mpc.CheckSwingHeightLin();
    mpc.CheckHolonomicLin();
    mpc.CheckCostFunctionDerivatives();
    // mpc.BenchmarkQuaternionIntegrationLin();
    // mpc.BenchmarkInverseDynamicsLin();
    // mpc.BenchmarkQuaternionConfigurationLin();
    // mpc.BenchmarkSwingHeightLin();
    // mpc.BenchmarkHolonomicLin();
    // mpc.BenchmarkConstraints();
    // mpc.BenchmarkCompute();
    mpc.BenchmarkCostFunctions();
}