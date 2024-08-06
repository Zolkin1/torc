//
// Created by zolkin on 6/20/24.
//

#include <eigen_utils.h>
#include <torc_timer.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "full_order_mpc.h"
#include "contact_schedule.h"

#define ENABLE_BENCHMARKS false

TEST_CASE("Basic MPC Test", "[mpc]") {
    using namespace torc::mpc;
    const std::string pin_model_name = "test_pin_model";
    std::filesystem::path a1_urdf = std::filesystem::current_path();
    a1_urdf += "/test_data/test_a1.urdf";

    std::filesystem::path mpc_config = std::filesystem::current_path();
    mpc_config += "/test_data/mpc_config.yaml";

    FullOrderMpc mpc(mpc_config, a1_urdf);

    mpc.Configure();

    torc::models::FullOrderRigidBody a1("a1", a1_urdf);

    vectorx_t random_state(a1.GetConfigDim() + a1.GetVelDim());
    // random_state << a1.GetNeutralConfig(), Eigen::VectorXd::Zero(a1.GetVelDim());
    // random_state(2) += 0.321;
    random_state << a1.GetRandomState();
    // random_state.tail(a1.GetVelDim()).setZero();
    std::cout << "initial config: " << random_state.head(a1.GetConfigDim()).transpose() << std::endl;
    std::cout << "initial vel: " << random_state.tail(a1.GetVelDim()).transpose() << std::endl;
    Trajectory traj;
    traj.UpdateSizes(a1.GetConfigDim(), a1.GetVelDim(), a1.GetNumInputs(), mpc.GetContactFrames(), mpc.GetNumNodes());

    vectorx_t q_neutral = a1.GetNeutralConfig();
    q_neutral(2) += 0.321;
    traj.SetDefault(q_neutral);

    mpc.SetWarmStartTrajectory(traj);

    ContactSchedule cs(mpc.GetContactFrames());
    const double contact_time = 0.3;
    double time = 0;
    for (int i = 0; i < 3; i++) {
        if (i % 2 != 0) {
            cs.InsertContact("FR_foot", time, time + contact_time);
            cs.InsertContact("RL_foot", time, time + contact_time);
        } else {
            cs.InsertContact("FL_foot", time, time + contact_time);
            cs.InsertContact("RR_foot", time, time + contact_time);
        }
        time += contact_time;
    }

    mpc.UpdateContactSchedule(cs);

    mpc.Compute(random_state, traj);

    // for (int i = 0; i < traj.GetNumNodes(); i++) {
    //     std::cout << "Node: " << i << std::endl;
    //     std::cout << "config: " << traj.GetConfiguration(i).transpose() << std::endl;
    //     std::cout << "vel: " << traj.GetVelocity(i).transpose() << std::endl;
    //     std::cout << "torque: " << traj.GetTau(i).transpose() << std::endl;
    // }

    // random_state = a1.GetRandomState();
    mpc.Compute(random_state, traj);

    mpc.PrintStatistics();
    std::cout << std::endl << std::endl;
    mpc.PrintContactSchedule();
}

TEST_CASE("Contact schedule", "[mpc][contact schedule]") {
    std::cout << "Contact Schedule Tests" << std::endl;
    torc::mpc::ContactSchedule cs({"LF_FRONT", "RF_FRONT"});
    cs.InsertContact("LF_FRONT", 0.1, 0.2);
    cs.InsertContact("RF_FRONT", 0.05, 0.15);

    CHECK(!cs.InContact("LF_FRONT", 0.075));
    CHECK(!cs.InContact("LF_FRONT", 0.25));
    CHECK(cs.InContact("LF_FRONT", 0.12));
    CHECK(cs.InContact("LF_FRONT", 0.2));

    CHECK(cs.InContact("RF_FRONT", 0.1));
    CHECK(cs.InContact("RF_FRONT", 0.12));
    CHECK(cs.InContact("RF_FRONT", 0.05));
    CHECK(!cs.InContact("RF_FRONT", 0.025));
    CHECK(!cs.InContact("RF_FRONT", 0.25));
}

#if ENABLE_BENCHMARKS
TEST_CASE("MPC Benchmarks [A1]", "[mpc][benchmarks]") {
    // Benchmarking with the A1
    using namespace torc::mpc;
    const std::string pin_model_name = "test_pin_model";
    std::filesystem::path a1_urdf = std::filesystem::current_path();
    a1_urdf += "/test_data/test_a1.urdf";

    std::filesystem::path mpc_config = std::filesystem::current_path();
    mpc_config += "/test_data/mpc_config.yaml";

    FullOrderMpc mpc(mpc_config, a1_urdf);
    mpc.SetVerbosity(false);

    BENCHMARK("MPC Configure [A1]") {
        mpc.Configure();
    };

    torc::models::FullOrderRigidBody a1("a1", a1_urdf);
    vectorx_t random_state = a1.GetRandomState();
    BENCHMARK("MPC Compute [A1]") {
        mpc.Compute(random_state);
    };
}
#endif


// using vector3_t = Eigen::Vector3d;
// using matrix3_t = Eigen::Matrix3d;
// using quat_t = Eigen::Quaterniond;
// using matrixx_t = Eigen::MatrixXd;
//
// matrix3_t DiffUpdateXi(const vector3_t& xi, const quat_t& q_bar, const quat_t& q_kp1, double dt, const vector3_t& w) {
//     matrix3_t Jlog = matrix3_t::Zero();
//     // matrixx_t Jexp = matrixx_t::Zero(4, 3);
//
//     // pinocchio::quaternion::Jexp3CoeffWise(xi, Jexp);
//     quat_t q_left = (q_kp1).inverse(); //*q_bar;
//     quat_t q_right = pinocchio::quaternion::exp3(w*dt);
//     CHECK(pinocchio::quaternion::log3(pinocchio::quaternion::exp3(xi)).isApprox(xi));
//     std::cout << "log(exp(xi)): " << pinocchio::quaternion::log3(pinocchio::quaternion::exp3(xi)).transpose() << std::endl;
//     pinocchio::quaternion::Jlog3(q_left*q_bar, Jlog);
//
//     return  Jlog; // * Jexp;
// }
//
// vector3_t Update(const vector3_t& xi, const quat_t& q_bar, const quat_t& q_kp1, double dt, const vector3_t& w) {
//     return pinocchio::quaternion::log3(
//         (q_kp1).inverse()
//         *q_bar*pinocchio::quaternion::exp3(xi)*pinocchio::quaternion::exp3(w*dt));
// }
//
// matrix3_t UpdateFdXi(const quat_t& q_kp1, const quat_t& q_k, const vector3_t& xi, const vector3_t& w_k) {
//     constexpr double DELTA = 1e-8;
//     double dt = 0.01;
//     const vector3_t update1 = Update(xi, q_k, q_kp1, dt, w_k);
//     vector3_t xi_ = xi;
//     matrix3_t update_fd = matrix3_t::Zero();
//     for (int i = 0; i < 3; i++) {
//         xi_(i) += DELTA;
//         vector3_t update2 = Update(xi_, q_k, q_kp1, dt, w_k);
//         for (int j = 0; j < 3; j++) {
//             update_fd(j, i) = (update2(j) - update1(j))/DELTA;
//         }
//
//         xi_(i) -= DELTA;
//     }
//
//     return update_fd;
// }
//
// TEST_CASE("Orientation Tests", "[mpc][orientation]") {
//     // Check exp derivatives
//
//     constexpr double DELTA = 1e-8;
//
//     std::cout << "===== Orientation Tests =====" << std::endl;
//     // exp
//     vector3_t r = vector3_t::Zero();
//     r.setRandom();
//     matrixx_t Jexp = matrixx_t::Zero(4, 3);
//     pinocchio::quaternion::Jexp3CoeffWise(r, Jexp);
//     std::cout << "Jexp: \n" << Jexp << std::endl;
//
//     quat_t q1;
//     vector3_t v = r;
//     pinocchio::quaternion::exp3(v, q1);
//
//     matrixx_t fd = matrixx_t::Zero(4, 3);
//     quat_t q2;
//     for (int i = 0; i < 3; i++) {
//         v(i) += DELTA;
//         pinocchio::quaternion::exp3(v, q2);
//         v(i) -= DELTA;
//
//         fd(0, i) = (q2.x() - q1.x())/DELTA;
//         fd(1, i) = (q2.y() - q1.y())/DELTA;
//         fd(2, i) = (q2.z() - q1.z())/DELTA;
//         fd(3, i) = (q2.w() - q1.w())/DELTA;
//     }
//     std::cout << "fd exp: \n" << fd << std::endl;
//
//     // log
//     q1 = pinocchio::quaternion::exp3(r);
//     matrix3_t Jlog = matrix3_t::Zero();
//     pinocchio::quaternion::Jlog3(q1, Jlog);
//     std::cout << "Jlog: \n" << Jlog << std::endl;
//
//     vector3_t v1 = pinocchio::quaternion::log3(q1);
//
//     matrix3_t fd_log = matrix3_t::Zero();
//     for (int i = 0; i < 3; i++) {
//         q2 = q1;
//         v.setZero();
//         v(i) += DELTA;
//         quat_t q_pert;
//         pinocchio::quaternion::exp3(v, q_pert);
//         q2 = q2*q_pert;
//
//         vector3_t v2 = pinocchio::quaternion::log3(q2);
//
//         for (int j = 0; j < 3; j++) {
//             fd_log(j, i) = (v2(j) - v1(j))/DELTA;
//         }
//     }
//     std::cout << "fd log: \n" << fd_log << std::endl;
//
//     std::cout << "----- qexp Derivative -----" << std::endl;
//     fd = matrixx_t::Zero(4, 3);
//
//     // Get a random quaternion
//     quat_t q_base;
//     r.setRandom();
//     r.setRandom();
//     q_base = pinocchio::quaternion::exp3(r);
//     v.setZero();
//     // v = r;
//     for (int i = 0; i < 3; i++) {
//         v(i) += DELTA;
//         pinocchio::quaternion::exp3(v, q2);
//         v(i) -= DELTA;
//         q2 = q_base*q2;
//
//         fd(0, i) = (q2.x() - q_base.x())/DELTA;
//         fd(1, i) = (q2.y() - q_base.y())/DELTA;
//         fd(2, i) = (q2.z() - q_base.z())/DELTA;
//         fd(3, i) = (q2.w() - q_base.w())/DELTA;
//     }
//     std::cout << "fd exp: \n" << fd << std::endl;
//
//     Jexp = matrixx_t::Zero(4, 3);
//     // JexpCoeffWise gives the variation wrt tangent space elements at the identity, but I want wrt tangent space elements at q_base
//     pinocchio::quaternion::Jexp3CoeffWise(pinocchio::quaternion::log3(q_base), Jexp);
//     std::cout << "Jexp: \n" << Jexp << std::endl;
//
//     std::cout << "----- Update Derivative -----" << std::endl;
//     // derivative of the update
//     vector3_t xi;
//     xi.setRandom();
//     std::cout << "xi: " << xi.transpose() << std::endl;
//
//     r.setRandom();
//     quat_t q_bar = pinocchio::quaternion::exp3(r);
//     vector3_t w_bar;
//     w_bar.setRandom();
//
//     double dt = 1;
//     quat_t q_kp1 = q_bar*pinocchio::quaternion::exp3(w_bar*dt);
//
//     vector3_t w;
//     // w.setZero();
//     w.setRandom();
//     // q_bar = q_bar*pinocchio::exp3(xi);
//     // xi.setZero();
//     // matrix3_t diff_an = DiffUpdateXi(xi, q_bar, q_kp1, dt, w);
//
//     // std::cout << "Analytical diff: \n" << diff_an << std::endl;
//
//     auto update_fd = UpdateFdXi(q_kp1, q_bar, xi, w);
//     // vector3_t update1 = Update(xi, q_bar, q_kp1, dt, w);
//     //
//     // matrix3_t update_fd = matrix3_t::Zero();
//     // for (int i = 0; i < 3; i++) {
//     //     xi(i) += DELTA;
//     //     for (int j = 0; j < 3; j++) {
//     //         update_fd(j, i) = (Update(xi, q_bar, q_kp1, dt, w)(j) - update1(j))/DELTA;
//     //     }
//     //
//     //     xi(i) -= DELTA;
//     // }
//
//     std::cout << "Finite diff: \n" << update_fd << std::endl;
//
//     BENCHMARK("update finite diff") {
//         auto update_fd = UpdateFdXi(q_kp1, q_bar, xi, w);
//     };
//
// #if ENABLE_BENCHMARKS
//     r.setRandom();
//     Jexp.setZero();
//     BENCHMARK("pinocchio Jexp") {
//         pinocchio::quaternion::Jexp3CoeffWise(r, Jexp);
//     };
//
//     q2 = pinocchio::quaternion::exp3(r);
//     matrix3_t Jlog = matrix3_t::Zero();
//     BENCHMARK("pinocchio Jlog") {
//         pinocchio::quaternion::Jlog3(q2, Jlog);
//     };
// #endif
// }