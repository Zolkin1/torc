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

    constexpr int NUM_CONFIGS = 10;
    for (int k = 0; k < NUM_CONFIGS; k++) {
        // Get a random configuration
        RobotState x_rand = model.GetRandomState();

        // Get random input
        vectorx_t input_rand = model.GetRandomInput();

        // Hold analytic derivatives
        matrixx_t A, B;
        A = matrixx_t::Zero(model.GetDerivativeDim(), model.GetDerivativeDim());
        B = matrixx_t::Zero(model.GetDerivativeDim(), model.GetInputDim());

        // Calculate analytic derivatives
        model.DynamicsDerivative(x_rand, input_rand, A, B);

        // Check wrt Configs
        RobotStateDerivative xdot_test = model.GetDynamics(x_rand, input_rand);
        for (int i = 0; i < x_rand.v.size(); i++) {
            RobotState x_d = x_rand;
            if (i >= 3 && i < 6) {
                // Need to take a step in lie algebra space
                Eigen::Vector3d v = Eigen::Vector3d::Zero();
                v(i - 3) += DELTA;

                x_d.q.array().segment<4>(3) = ((x_d.Quat() * pinocchio::quaternion::exp3(v)).coeffs());
                continue;
            } else {
                if (i >= 6) {
                    x_d.q(i + 1) += DELTA;
                } else {
                    x_d.q(i) += DELTA;
                }
            }
            RobotStateDerivative deriv_d = model.GetDynamics(x_d, input_rand);

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (deriv_d.v(j) - xdot_test.v(j)) / DELTA;
                REQUIRE_THAT(fd - A(j, i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (deriv_d.a(j) - xdot_test.a(j)) / DELTA;
                REQUIRE_THAT(fd - A(j + x_rand.v.size(), i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }
        }

        // Now check wrt velocities
        for (int i = 0; i < x_rand.v.size(); i++) {
            RobotState x_d = x_rand;

            x_d.v(i) += DELTA;

            RobotStateDerivative deriv_d = model.GetDynamics(x_d, input_rand);

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (deriv_d.v(j) - xdot_test.v(j)) / DELTA;
                REQUIRE_THAT(fd - A(j, i + x_rand.v.size()), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (deriv_d.a(j) - xdot_test.a(j)) / DELTA;
                REQUIRE_THAT(fd - A(j + x_rand.v.size(), i + x_rand.v.size()), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }
        }

        // Now check wrt inputs
        for (int i = 0; i < input_rand.size(); i++) {
            vectorx_t input_d = input_rand;

            input_d(i) += DELTA;

            RobotStateDerivative deriv_d = model.GetDynamics(x_rand, input_d);

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (deriv_d.v(j) - xdot_test.v(j)) / DELTA;
                REQUIRE_THAT(fd - B(j, i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (deriv_d.a(j) - xdot_test.a(j)) / DELTA;
                REQUIRE_THAT(fd - B(j + x_rand.v.size(), i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }
        }
    }
}

TEST_CASE("Basic Centroidal Model Test", "[model][centroidal]") {
    using namespace torc::models;
    auto integrator = Centroid("intgerator", "test_data/integrator.urdf", {}, {});
    auto &model = integrator;

    CheckDerivatives(integrator);

    // // Get a random configuration
    // RobotState x_rand = model.GetRandomState();
    //
    // // Get random input
    // vectorx_t input_rand = model.GetRandomInput();
    //
    // // Hold analytic derivatives
    // matrixx_t A = matrixx_t::Zero(model.GetDerivativeDim(), model.GetDerivativeDim());
    // matrixx_t B = matrixx_t::Zero(model.GetDerivativeDim(), model.GetInputDim());
    //
    // // Calculate analytic derivatives
    // model.GetDynamics(x_rand, input_rand);
    // model.DynamicsDerivative(x_rand, input_rand, A, B);
}