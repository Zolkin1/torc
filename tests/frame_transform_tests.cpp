//
// Created by zolkin on 2/16/25.
//
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "full_order_rigid_body.h"

TEST_CASE("Pose Test", "[Frames]") {
    using namespace torc::models;

    // Load in the model
    std::filesystem::path g1_urdf = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/g1_hand_v2.urdf";
    torc::models::FullOrderRigidBody g1("g1", g1_urdf);

    for (int i = 0; i < 10; i++) {
        // Choose a random configuration
        vectorx_t q_rand = g1.GetRandomConfig();

        // Get frame pose in the world
        g1.FirstOrderFK(q_rand);
        pinocchio::SE3 frame_pose = g1.GetFrameState("left_toe").placement;

        // Try to get the floating base position given this frame and the joint config
        vectorx_t q_guess = q_rand;
        q_guess.head<3>().setZero();
        q_guess.segment<4>(3) << 0, 0, 0, 1;

        pinocchio::SE3 floating_base_pose = g1.TransformPose(frame_pose, "left_toe", "pelvis", q_guess);

        // Check result
        CHECK(floating_base_pose.translation().isApprox(q_rand.head<3>()));
        quat_t quat_res(floating_base_pose.rotation());
        if (quat_res.x() >= 0 && q_rand(3) >= 0) {
            CHECK(quat_res.isApprox(quat_t( q_rand.segment<4>(3))));
        } else {
            CHECK(quat_res.isApprox(quat_t( -q_rand.segment<4>(3))));
        }


        std::cout << "res pos: " << floating_base_pose.translation().transpose() << std::endl;
        std::cout << "res quat: " << quat_res << std::endl;
        std::cout << "actual pos: " << q_rand.head<3>().transpose() << std::endl;
        std::cout << "actual quat: " << q_rand.segment<4>(3).transpose() << std::endl;
    }
}

// TEST_CASE("Velocity Test", "[Frames]") {
//     using namespace torc::models;
//
//     // Load in the model
//     std::filesystem::path g1_urdf = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/g1_hand_v2.urdf";
//     torc::models::FullOrderRigidBody g1("g1", g1_urdf);
//
//     // for (int i = 0; i < 10; i++) {
//     // Choose a random configuration
//     vectorx_t q_rand = g1.GetRandomConfig();
//
//     // Choose a random velocity
//     vectorx_t v_rand = g1.GetRandomVel();
//
//     // Get frame vel in the local frame
//     g1.SecondOrderFK(q_rand, v_rand);
//     pinocchio::Motion frame_vel = g1.GetFrameState("left_toe", pinocchio::LOCAL).vel;
//
//     // Try to get the velocity of another frame given the robot's configuration and velocity of the current frame
//     pinocchio::Motion pelvis_vel = g1.TransformVelocity(frame_vel, "left_toe", "pelvis", q_rand);
//
//     // Check result
//     CHECK(pelvis_vel.linear().isApprox(v_rand.head<3>()));
//     CHECK(pelvis_vel.angular().isApprox(v_rand.segment<3>(3)));
//
//
//     std::cout << "res linear vel: " << pelvis_vel.linear().transpose() << std::endl;
//     std::cout << "res angular vel: " << pelvis_vel.angular().transpose() << std::endl;
//     std::cout << "actual linear vel: " << v_rand.head<3>().transpose() << std::endl;
//     std::cout << "actual anguarl vel: " << v_rand.segment<3>(3).transpose() << std::endl;
//     // }
// }