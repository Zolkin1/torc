// #include <catch2/catch_test_macros.hpp>
// #include <catch2/matchers/catch_matchers_floating_point.hpp>
// #include <catch2/catch_get_random_seed.hpp>
// #include <catch2/generators/catch_generators.hpp>
// #include <catch2/benchmark/catch_benchmark.hpp>
//
// #include "centroidal_model.h"
// #include <iostream>
//
// void CheckDerivatives(torc::models::Centroid& model) {
//     using namespace torc::models;
//
//     constexpr double FD_MARGIN = 1e-3;
//     constexpr double DELTA = 1e-8;
//
//     // Checking derivatives with finite differences
//     srand(Catch::getSeed());        // Set the srand seed manually
//
//     constexpr int N_CONFIGS = 6;
//     for (int k = 0; k < N_CONFIGS; k++) {
//         RobotState x_state = model.GetRandomState();
//         vectorx_t input = model.GetRandomInput();
//
//         const int nq = x_state.q_dim();
//         const int nv = x_state.v_dim();
//         const int nu = input.size();
//         const int quat_start_idx = nv+3;
//         const int quat_end_idx = nv+7;
//
//         matrixx_t A, B;
//         model.DynamicsDerivative(x_state, input, A, B);
//
//         // test derivatives with respect to the state
//         for (int i=0; i<nq+nv; i++) {
//             RobotState x_state_new = x_state;
//             if (i < nv) {
//                 // hcom
//                 vectorx_t delta_x_v = vectorx_t::Zero(nv);
//                 delta_x_v(i) = DELTA;
//                 x_state_new = RobotState(x_state.q, x_state.v + delta_x_v);
//             } else if (i >= quat_start_idx && i < quat_end_idx) {
//                 if (i+1 == quat_end_idx) {
//                     // since i is the state vector index, it matches the quaternion indicies in four places. However,
//                     // the quaternion's tangent space is only three dimensional, so we skip the last match.
//                     continue;
//                 }
//                 // base orientation (quaternion)
//                 Eigen::Vector3d v = Eigen::Vector3d::Zero();
//                 v(i - quat_start_idx) += DELTA;  // take a step in tangent space
//                 x_state_new.q.array().segment<4>(3) = (x_state_new.Quat() * pinocchio::quaternion::exp3(v)).coeffs();
//             } else {
//                 // base position OR joint positions
//                 vectorx_t delta_x_q = vectorx_t::Zero(nq);
//                 delta_x_q(i - nv) = DELTA;
//                 x_state_new = RobotState(x_state.q + delta_x_q, x_state.v);
//             }
//
//             // calculate derivative from finite difference
//             RobotStateDerivative dx_0 = model.GetDynamics(x_state, input);
//             RobotStateDerivative dx_1 = model.GetDynamics(x_state_new, input);
//             vectorx_t dv_dx_numerical = (dx_1.v - dx_0.v) / DELTA;
//             vectorx_t da_dx_numerical = (dx_1.a - dx_0.a) / DELTA;
//
//             // calculate derivative from analytical solution
//
//             std::cout << "index = " << i << std::endl;
//             std::cout << A.block(0, 0, Centroid::COM_DOF, A.cols()).col(i) << "\n------\n";
//             std::cout << da_dx_numerical << "\n------\n" << std::flush;
//
//             // REQUIRE(dv_dx_numerical.isApprox(delta_v_analytical, FD_MARGIN));
//             REQUIRE(da_dx_numerical.isApprox(delta_a_analytical, FD_MARGIN));
//         }
//
//         // test derivatives wrt the inputs
//         for (int i=0; i<nu; i++) {
//             vectorx_t input_new = input;
//             input_new(i) += DELTA;
//             // calculate derivative from finite difference
//             RobotStateDerivative dx_0 = model.GetDynamics(x_state, input);
//             RobotStateDerivative dx_1 = model.GetDynamics(x_state, input_new);
//             vectorx_t delta_v_numerical = dx_1.v - dx_0.v;
//             vectorx_t delta_a_numerical = dx_1.a - dx_0.a;
//
//             // calculate derivative from analytical solution
//             vectorx_t delta_dx_analytical = B * input_new;
//             vectorx_t delta_v_analytical = delta_dx_analytical.segment(Centroid::COM_DOF, nv);
//             vectorx_t delta_a_analytical = delta_dx_analytical.segment(0, Centroid::COM_DOF);
//
//             std::cout << "index = " << i << std::endl;
//             std::cout << delta_a_analytical << "\n------\n";
//             std::cout << delta_a_numerical << "\n------\n" << std::flush;
//
//             // REQUIRE(delta_v_numerical.isApprox(delta_v_analytical, FD_MARGIN));
//             REQUIRE(delta_a_numerical.isApprox(delta_a_analytical, FD_MARGIN));
//         }
//     }
// }
//
// TEST_CASE("Basic Centroidal Model Test", "[model][centroidal]") {
//     using namespace torc::models;
//     auto integrator = Centroid("intgerator", "test_data/hopper.urdf", {"foot"}, {});
//     auto &model = integrator;
//     CheckDerivatives(integrator);
// }