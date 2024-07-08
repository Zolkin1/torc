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
    constexpr double DELTA = 1e-5;

    // Checking derivatives with finite differences
    // srand(Catch::getSeed());        // Set the srand seed manually
    srand(0);        // Set the srand seed manually

    constexpr int N_CONFIGS = 6;
    for (int k = 0; k < N_CONFIGS; k++) {
        vectorx_t x0 = model.GetRandomState();
        vectorx_t q0, h0;
        model.ParseState(x0, q0, h0);
        vectorx_t input = model.GetRandomInput();

        constexpr size_t nv = Centroid::COM_DOF;
        constexpr size_t quat_start_idx = 3;
        constexpr size_t quat_end_idx = 7;

        matrixx_t A, B;
        model.DynamicsDerivative(x0, input, A, B);
        const size_t ndx = A.rows();

        // test derivatives with respect to the state
        for (int i=0; i<ndx; i++) {
            vectorx_t x1;
            vectorx_t h1 = h0;
            vectorx_t q1 = q0;
            if (i < q0.size()) {
                if (i >= quat_start_idx && i < quat_end_idx) {
                    if (i+1 == quat_end_idx) {
                        // since i is the state vector index, it matches the quaternion indicies in four places. However,
                        // the quaternion's tangent space is only three dimensional, so we skip the last match.
                        continue;
                    }
                    // base orientation (quaternion)
                    vector3_t d_quat = vector3_t::Zero();
                    d_quat(i - quat_start_idx) += DELTA;  // take a step in tangent space
                    q1.array().segment<4>(3) = (Centroid::ParseBaseOrientation(q0) * pinocchio::quaternion::exp3(d_quat)).coeffs();
                    x1 = Centroid::BuildState(q1, h0);
                } else {
                    q1(i) += DELTA;
                    x1 = Centroid::BuildState(q1, h0);
                }
            } else {
                h1(i - q0.size()) += DELTA;
                x1 = Centroid::BuildState(q0, h1);
            }

            // calculate derivative from finite difference
            vectorx_t dx0 = model.GetDynamics(x0, input);
            vectorx_t dx1 = model.GetDynamics(x1, input);
            // std::cout << "x0\n";
            // std::cout << x0 << std::endl;
            // std::cout << "----\n";
            // std::cout << "x1\n";
            // std::cout << x1 << std::endl;
            // std::cout << "----\n";
            vectorx_t ddx_dx_numerical = (dx1 - dx0) / DELTA;

            // calculate derivative from analytical solution
            vectorx_t ddx_dx_analytical = A.col(i);

            std::cout << "index = " << i << "\n-----\n" << std::endl;
            std::cout << "analytical\n" << ddx_dx_analytical << "\n------\n" << std::endl;
            std::cout << "numerical\n" << ddx_dx_numerical << "\n------\n" << std::endl;

            REQUIRE((ddx_dx_analytical.bottomRows(ndx-6) - ddx_dx_numerical.bottomRows(ndx-6)).norm() < FD_MARGIN);
        }

        // // test derivatives wrt the inputs
        // for (int i=0; i<nu; i++) {
        //     vectorx_t input_new = input;
        //     input_new(i) += DELTA;
        //     // calculate derivative from finite difference
        //     RobotStateDerivative dx_0 = model.GetDynamics(x, input);
        //     RobotStateDerivative dx_1 = model.GetDynamics(x, input_new);
        //     vectorx_t delta_v_numerical = dx_1.v - dx_0.v;
        //     vectorx_t delta_a_numerical = dx_1.a - dx_0.a;
        //
        //     // calculate derivative from analytical solution
        //     vectorx_t delta_dx_analytical = B * input_new;
        //     vectorx_t delta_v_analytical = delta_dx_analytical.segment(Centroid::COM_DOF, nv);
        //     vectorx_t delta_a_analytical = delta_dx_analytical.segment(0, Centroid::COM_DOF);
        //
        //     std::cout << "index = " << i << std::endl;
        //     std::cout << delta_a_analytical << "\n------\n";
        //     std::cout << delta_a_numerical << "\n------\n" << std::flush;
        //
        //     // REQUIRE(delta_v_numerical.isApprox(delta_v_analytical, FD_MARGIN));
        //     REQUIRE(delta_a_numerical.isApprox(delta_a_analytical, FD_MARGIN));
        // }
    }
}

TEST_CASE("Basic Centroidal Model Test", "[model][centroid]") {
    std::cout.precision(12);
    using namespace torc::models;
    auto model = Centroid("hopper", "test_data/hopper.urdf", {"foot"}, {});
    CheckDerivatives(model);
}