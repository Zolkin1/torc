#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "centroidal_model.h"
#include <iostream>

void CheckDerivatives(torc::models::Centroid& model) {
    using namespace torc::models;
    constexpr double FD_MARGIN = 1e-3;
    constexpr double DELTA = 1e-8;

    // Checking derivatives with finite differences
    srand(Catch::getSeed());        // Set the srand seed manually

    constexpr int N_CONFIGS = 6;
    for (int k = 0; k < N_CONFIGS; k++) {
        RobotState x_rand = model.GetRandomState();
        vectorx_t u_rand = model.GetRandomInput();

        matrixx_t A = matrixx_t::Zero(model.GetDerivativeDim(), model.GetDerivativeDim());
        matrixx_t B = matrixx_t::Zero(model.GetDerivativeDim(), model.GetInputDim());

        model.DynamicsDerivative(x_rand, u_rand, A, B);

        vectorx_t delta_x_v = vectorx_t::Zero(6);
        delta_x_v(k) = DELTA;

        RobotStateDerivative dx_0 = model.GetDynamics(x_rand, u_rand);
        x_rand.v += delta_x_v;
        RobotStateDerivative dx_1 = model.GetDynamics(x_rand, u_rand);

        vectorx_t delta_v_numerical = dx_1.v - dx_0.v;
        vectorx_t delta_a_numerical = dx_1.a - dx_0.a;

        vectorx_t delta_x(model.GetStateDim());
        delta_x << delta_x_v, vectorx_t::Zero(7);
        vectorx_t delta_dx_analytical = A * delta_x;
        vectorx_t delta_v_analytical = delta_dx_analytical.segment(Centroid::COM_DOF, Centroid::COM_DOF);
        vectorx_t delta_a_analytical = delta_dx_analytical.segment(0, Centroid::COM_DOF);

        REQUIRE(delta_v_numerical.isApprox(delta_v_analytical, FD_MARGIN));
        REQUIRE(delta_a_numerical.isApprox(delta_a_analytical, FD_MARGIN));

        // std::cout << delta_v_analytical << "\n------\n";
        // std::cout << delta_v_analytical;
    }
}

TEST_CASE("Basic Centroidal Model Test", "[model][centroidal]") {
    using namespace torc::models;
    auto integrator = Centroid("intgerator", "test_data/integrator.urdf", {}, {});
    auto &model = integrator;
    CheckDerivatives(integrator);
}