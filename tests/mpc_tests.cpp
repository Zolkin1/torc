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

TEST_CASE("A1 MPC Test", "[mpc]") {
    using namespace torc::mpc;
    const std::string pin_model_name = "test_pin_model";
    std::filesystem::path a1_urdf = std::filesystem::current_path();
    a1_urdf += "/test_data/test_a1.urdf";

    std::filesystem::path mpc_config = std::filesystem::current_path();
    mpc_config += "/test_data/mpc_config.yaml";

    FullOrderMpc mpc("a1_mpc", mpc_config, a1_urdf);

    mpc.Configure();

    torc::models::FullOrderRigidBody a1("a1", a1_urdf);

    vectorx_t random_state(a1.GetConfigDim() + a1.GetVelDim());
    vectorx_t q_rand = a1.GetRandomConfig();
    vectorx_t v_rand = a1.GetRandomVel();
    // random_state.tail(a1.GetVelDim()).setZero();
    std::cout << "initial config: " << q_rand.transpose() << std::endl;
    std::cout << "initial vel: " << v_rand.transpose() << std::endl;
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

    mpc.Compute(q_rand, v_rand, traj);

    // for (int i = 0; i < traj.GetNumNodes(); i++) {
    //     std::cout << "Node: " << i << std::endl;
    //     std::cout << "config: " << traj.GetConfiguration(i).transpose() << std::endl;
    //     std::cout << "vel: " << traj.GetVelocity(i).transpose() << std::endl;
    //     std::cout << "torque: " << traj.GetTau(i).transpose() << std::endl;
    // }

    // random_state = a1.GetRandomState();
    mpc.Compute(q_rand, v_rand, traj);

    mpc.PrintStatistics();
    std::cout << std::endl << std::endl;
    mpc.PrintContactSchedule();
    std::cout << std::endl << std::endl;
}

TEST_CASE("Achilles MPC Test", "[mpc]") {
    using namespace torc::mpc;
    const std::string pin_model_name = "test_pin_model";
    std::filesystem::path achilles_urdf = std::filesystem::current_path();
    achilles_urdf += "/test_data/achilles.urdf";

    std::filesystem::path mpc_config = std::filesystem::current_path();
    mpc_config += "/test_data/achilles_mpc_config.yaml";

    FullOrderMpc mpc("achilles_mpc", mpc_config, achilles_urdf);

    mpc.Configure();

    torc::models::FullOrderRigidBody achilles("achilles", achilles_urdf);

    vectorx_t random_state(achilles.GetConfigDim() + achilles.GetVelDim());
    vectorx_t q_rand = achilles.GetRandomConfig();
    vectorx_t v_rand = achilles.GetRandomVel();

    // random_state.tail(a1.GetVelDim()).setZero();
    std::cout << "initial config: " << random_state.head(achilles.GetConfigDim()).transpose() << std::endl;
    std::cout << "initial vel: " << random_state.tail(achilles.GetVelDim()).transpose() << std::endl;

    ContactSchedule cs(mpc.GetContactFrames());
    const double contact_time = 0.3;
    double time = 0;
    for (int i = 0; i < 3; i++) {
        if (i % 2 != 0) {
            cs.InsertContact("right_foot", time, time + contact_time);
            // cs.InsertContact("right_hand", time, time + contact_time);
        } else {
            cs.InsertContact("left_foot", time, time + contact_time);
            // cs.InsertContact("left_hand", time, time + contact_time);
        }
        time += contact_time;
    }

    mpc.UpdateContactSchedule(cs);

    vectorx_t q_target;
    q_target.resize(achilles.GetConfigDim());
    q_target << 0, 0, 0.97,
                0, 0, 0, 1,
                0, 0, -0.26,
                0, 0.65, -0.43,
                0, 0, 0,
                0, 0, -0.26,
                0.65, -0.43,
                0, 0, 0;
    mpc.SetConstantConfigTarget(q_target);

    Trajectory traj;
    traj.UpdateSizes(achilles.GetConfigDim(), achilles.GetVelDim(), achilles.GetNumInputs(), mpc.GetContactFrames(), mpc.GetNumNodes());

    vectorx_t q_neutral = achilles.GetNeutralConfig();
    q_neutral(2) += 0.321;
    traj.SetDefault(q_neutral);

    mpc.SetWarmStartTrajectory(traj);

    mpc.Compute(q_rand, v_rand, traj);

    // for (int i = 0; i < traj.GetNumNodes(); i++) {
    //     std::cout << "Node: " << i << std::endl;
    //     std::cout << "config: " << traj.GetConfiguration(i).transpose() << std::endl;
    //     std::cout << "vel: " << traj.GetVelocity(i).transpose() << std::endl;
    //     std::cout << "torque: " << traj.GetTau(i).transpose() << std::endl;
    // }

    mpc.Compute(q_rand, v_rand, traj);

    mpc.PrintStatistics();
    std::cout << std::endl << std::endl;
    mpc.PrintContactSchedule();
    std::cout << std::endl << std::endl;
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

TEST_CASE("Trajectory Interpolation", "[trajectory]") {
    std::cout << "====== Trajectory Interpolation =====" << std::endl;
    using namespace torc::mpc;

    std::filesystem::path achilles_urdf = std::filesystem::current_path();
    achilles_urdf += "/test_data/achilles.urdf";
    torc::models::FullOrderRigidBody achilles("achilles", achilles_urdf);

    std::filesystem::path mpc_config = std::filesystem::current_path();
    mpc_config += "/test_data/achilles_mpc_config.yaml";

    FullOrderMpc mpc("achilles_mpc", mpc_config, achilles_urdf);

    Trajectory traj;
    traj.UpdateSizes(achilles.GetConfigDim(), achilles.GetVelDim(), achilles.GetNumInputs(),
        mpc.GetContactFrames(), mpc.GetNumNodes());

    vectorx_t q_neutral = achilles.GetNeutralConfig();
    traj.SetDefault(q_neutral);

    std::vector<double> dt(mpc.GetNumNodes() - 1);
    std::fill(dt.begin(), dt.end(), 0.02);
    traj.SetDtVector(dt);

    // ----- Configuration ----- //
    // Checks when all the configurations are the same
    vectorx_t q_interp;
    traj.GetConfigInterp(0, q_interp);
    CHECK(q_interp.isApprox(q_neutral));

    traj.GetConfigInterp(0.1, q_interp);
    CHECK(q_interp.isApprox(q_neutral));

    traj.GetConfigInterp(0.213, q_interp);
    CHECK(q_interp.isApprox(q_neutral));

    // Checks when there are differences
    vectorx_t q_rand = achilles.GetRandomConfig();
    traj.SetConfiguration(1, q_rand);

    traj.GetConfigInterp(0, q_interp);
    CHECK(q_interp.isApprox(q_neutral));

    traj.GetConfigInterp(0.01, q_interp);
    // This is the expected vector EXCEPT for the quaternion part
    vectorx_t q_expect = 0.5*q_rand + 0.5*q_neutral;

    CHECK(q_interp.head<3>().isApprox(q_expect.head<3>()));
    CHECK(q_interp.tail(achilles.GetNumInputs()).isApprox(q_expect.tail(achilles.GetNumInputs())));
    // TODO: Add check for quaternion values

    // ----- Velocity ----- //
    // Checks when all the configurations are the same
    vectorx_t v_zero = vectorx_t::Zero(achilles.GetVelDim());
    vectorx_t v_interp;
    traj.GetVelocityInterp(0, v_interp);
    CHECK(v_interp.isApprox(v_zero));

    traj.GetVelocityInterp(0.1, v_interp);
    CHECK(v_interp.isApprox(v_zero));

    traj.GetVelocityInterp(0.213, v_interp);
    CHECK(v_interp.isApprox(v_zero));

    // Checks when there are differences
    vectorx_t v_rand = achilles.GetRandomVel();
    traj.SetVelocity(1, v_rand);

    traj.GetVelocityInterp(0, v_interp);
    CHECK(v_interp.isApprox(v_zero));

    traj.GetVelocityInterp(0.01, v_interp);
    vectorx_t v_expect = 0.5*v_rand + 0.5*v_zero;
    CHECK(v_interp.isApprox(v_expect));

    traj.GetVelocityInterp(0.005, v_interp);
    v_expect = 0.25*v_rand + 0.75*v_zero;
    CHECK(v_interp.isApprox(v_expect));

    // ---- Torque ----- //
    // (Mostly covered by velocity because they use the same internals)
    vectorx_t tau_interp;
    vectorx_t tau_zero = vectorx_t::Zero(achilles.GetNumInputs());

    vectorx_t tau_rand = vectorx_t::Random(achilles.GetNumInputs());
    traj.SetTau(1, tau_rand);

    traj.GetTorqueInterp(0.01, tau_interp);
    vectorx_t tau_expect = 0.5*tau_rand + 0.5*tau_zero;
    CHECK(tau_interp.isApprox(tau_expect));

    // TODO: Add force checks
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