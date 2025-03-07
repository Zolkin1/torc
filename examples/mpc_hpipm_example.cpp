//
// Created by zolkin on 1/18/25.
//

#include <torc_timer.h>
#include <pthread.h>

#include "CentroidalDynamicsConstraint.h"
#include "DynamicsConstraint.h"
#include "ForwardKinematicsCost.h"
#include "hpipm_mpc.h"
#include "SRBConstraint.h"

void thread_function() {
    // struct sched_param param;
    // param.sched_priority = 99;
    // pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

    // Prevents uncessary cache misses
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(2, &cpuset);  // Pin to CPU 2

    pthread_t thread = pthread_self();
    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_setaffinity_np");
    }

    // Create all the constraints
    using namespace torc::mpc;
    std::filesystem::path g1_urdf = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/g1_hand_v2.urdf";

    std::filesystem::path mpc_config = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/g1_mpc_config_2.yaml";

    MpcSettings settings(mpc_config);
    settings.Print();

    std::vector<std::pair<std::string, std::string>> poly_contact_frames;
    poly_contact_frames.emplace_back(settings.polytope_frames[0], settings.contact_frames[0]);
    poly_contact_frames.emplace_back(settings.polytope_frames[0], settings.contact_frames[1]);
    poly_contact_frames.emplace_back(settings.polytope_frames[1], settings.contact_frames[2]);
    poly_contact_frames.emplace_back(settings.polytope_frames[1], settings.contact_frames[3]);

    settings.poly_contact_pairs = poly_contact_frames;

    const std::string pin_model_name = "test_pin_model";

    torc::models::FullOrderRigidBody g1("g1", g1_urdf, settings.joint_skip_names, settings.joint_skip_values);

    fs::path deriv_lib_path = fs::current_path();
    deriv_lib_path = deriv_lib_path / "deriv_libs";

    // --------------------------------- //
    // ---------- Constraints ---------- //
    // --------------------------------- //
    // ---------- FO Dynamics ---------- //
    DynamicsConstraint dynamics_constraints(g1, settings.contact_frames, "g1_full_order",
        deriv_lib_path, settings.compile_derivs, 0, settings.nodes_full_dynamics);

    // ---------- Centroidal Dynamics ---------- //
    CentroidalDynamicsConstraint centroidal_dynamics(g1, settings.contact_frames, "g1_centroidal", settings.deriv_lib_path,
        settings.compile_derivs, settings.nodes_full_dynamics, settings.nodes);

    // ---------- SRB Dynamics ---------- //
    SRBConstraint srb_dynamics(settings.nodes_full_dynamics, settings.nodes,
        "g1_srb",
        settings.contact_frames, settings.deriv_lib_path, settings.compile_derivs, g1, settings.q_target);

    // ---------- Box Constraints ---------- //
    // Config
    std::vector<int> config_lims_idxs;
    for (int i = 0; i < g1.GetConfigDim() - torc::mpc::FLOATING_BASE; ++i) {
        config_lims_idxs.push_back(i + FLOATING_VEL);
    }
    BoxConstraint config_box(1, settings.nodes, "config_box",
        g1.GetLowerConfigLimits().tail(g1.GetConfigDim() - torc::mpc::FLOATING_BASE),
        g1.GetUpperConfigLimits().tail(g1.GetConfigDim() - torc::mpc::FLOATING_BASE),
        config_lims_idxs);

    // Vel
    std::vector<int> vel_lims_idxs;
    for (int i = 0; i < g1.GetVelDim() - torc::mpc::FLOATING_VEL; ++i) {
        vel_lims_idxs.push_back(i + FLOATING_VEL);
    }
    std::cout << g1.GetVelocityJointLimits().transpose() << std::endl;
    BoxConstraint vel_box(1, settings.nodes, "vel_box",
        -g1.GetVelocityJointLimits().tail(g1.GetVelDim() - torc::mpc::FLOATING_VEL),
        g1.GetVelocityJointLimits().tail(g1.GetVelDim() - torc::mpc::FLOATING_VEL),
        vel_lims_idxs);

    // Torque
    std::vector<int> tau_lims_idxs;
    for (int i = 0; i < g1.GetVelDim() - torc::mpc::FLOATING_VEL; ++i) {
        tau_lims_idxs.push_back(i);
    }
    BoxConstraint tau_box(0, settings.nodes, "tau_box",
        -g1.GetTorqueJointLimits().tail(g1.GetVelDim() - torc::mpc::FLOATING_VEL),
        g1.GetTorqueJointLimits().tail(g1.GetVelDim() - torc::mpc::FLOATING_VEL),
        tau_lims_idxs);

    // Force
    std::vector<int> force_lim_idxs;
    for (int i = 0; i < 3; i++) {
        force_lim_idxs.push_back(i);
    }
    vectorx_t stance_lb(3), stance_ub(3);
    stance_lb << -1000, -1000, settings.min_grf;
    stance_ub << 1000, 1000, settings.max_grf;
    BoxConstraint stance_force_box(0, settings.nodes, "force_box",
        stance_lb, // Minimum force on the ground
        stance_ub,
        force_lim_idxs);

    vectorx_t swing_lb(3), swing_ub(3);
    swing_lb << 0, 0, 0;
    swing_ub << 0, 0, 0;
    BoxConstraint swing_force_box(0, settings.nodes, "force_box",
        swing_lb, // Minimum force on the ground
        swing_ub,
        force_lim_idxs);

    // ---------- Friction Cone Constraints ---------- //
    FrictionConeConstraint friction_cone_constraint(0,settings.nodes - 1, "friction_cone_cone",
        settings.friction_coef, settings.friction_margin, settings.deriv_lib_path, settings.compile_derivs);

    // ---------- Swing Constraints ---------- //
    SwingConstraint swing_constraint(settings.swing_start_node, settings.swing_end_node,
        "swing_constraint", g1, settings.contact_frames, settings.deriv_lib_path, settings.compile_derivs);

    // ---------- Holonomic Constraints ---------- //
    HolonomicConstraint holonomic_constraint(settings.holonomic_start_node, settings.holonomic_end_node,
        "holonomic_constraint", g1, settings.contact_frames, settings.deriv_lib_path, settings.compile_derivs);

    // ---------- Collision Constraints ---------- //
    CollisionConstraint collision_constraint(settings.collision_start_node, settings.collision_end_node,
        "collision_constraint", g1, settings.deriv_lib_path, settings.compile_derivs, settings.collision_data);

    // ---------- Polytope Constraints ---------- //
    PolytopeConstraint polytope_constraint(settings.polytope_start_node, settings.polytope_end_node, "polytope_constraint",
        settings.polytope_frames, settings.deriv_lib_path, settings.compile_derivs, g1);

    std::cout << "===== Constraints Created =====" << std::endl;

    // --------------------------------- //
    // ------------- Costs ------------- //
    // --------------------------------- //
    // ---------- Velocity Tracking ---------- //
    std::cout << settings.cost_data.at(1).weight.transpose() << std::endl;
    LinearLsCost vel_tracking(0, settings.nodes, "vel_tracking",
        settings.deriv_lib_path, settings.compile_derivs, settings.cost_data.at(1).weight.size());

    // ---------- Tau Tracking ---------- //
    LinearLsCost tau_tracking(0, settings.nodes, "tau_tracking",
        settings.deriv_lib_path, settings.compile_derivs, settings.cost_data.at(2).weight.size());

    // ---------- Force Tracking ---------- //
    LinearLsCost force_tracking(0, settings.nodes, "force_tracking",
        settings.deriv_lib_path, settings.compile_derivs, settings.cost_data.at(3).weight.size());

    // ---------- Config Tracking ---------- //
    ConfigTrackingCost config_tracking(0, settings.nodes, "config_tracking", settings.cost_data.at(0).weight.size(),
        settings.deriv_lib_path, settings.compile_derivs, g1);

    // ---------- Forward Kinematics Tracking ---------- //
    // For now they all need the same weight
    ForwardKinematicsCost fk_cost(0, settings.nodes, "fk_cost", settings.cost_data.at(4).weight.size(),
        settings.deriv_lib_path, settings.compile_derivs, g1, settings.contact_frames);

    // --------------------------------- //
    // ------- Contact Schedule -------- //
    // --------------------------------- //
    torc::mpc::ContactSchedule cs(settings.contact_frames);

    cs.InsertSwing("right_toe", 0.1, 0.5);
    cs.InsertSwing("right_heel", 0.1, 0.5);
    cs.InsertSwing("left_toe", 0.5, 0.9);
    cs.InsertSwing("left_heel", 0.5, 0.9);

    cs.InsertSwing("right_toe", 0.9, 1.2);
    cs.InsertSwing("right_heel", 0.9, 1.2);
    cs.InsertSwing("left_toe", 1.2, 1.6);
    cs.InsertSwing("left_heel", 1.2, 1.6);

    cs.InsertSwing("right_toe", 1.6, 2.0);
    cs.InsertSwing("right_heel", 1.6, 2.0);
    cs.InsertSwing("left_toe", 2.0, 2.4);
    cs.InsertSwing("left_heel", 2.0, 2.4);

    cs.InsertSwing("right_toe", 2.4, 2.8);
    cs.InsertSwing("right_heel", 2.4, 2.8);
    cs.InsertSwing("left_toe", 2.8, 3.2);
    cs.InsertSwing("left_heel", 2.8, 3.2);
    // --------------------------------- //
    // -------------- MPC -------------- //
    // --------------------------------- //
    HpipmMpc mpc(settings, g1);

    std::cout << "===== MPC Created =====" << std::endl;
    mpc.SetDynamicsConstraints(std::move(dynamics_constraints));
    mpc.SetCentroidalDynamicsConstraints(std::move(centroidal_dynamics));
    // mpc.SetSrbConstraint(std::move(srb_dynamics));
    mpc.SetConfigBox(config_box);
    mpc.SetVelBox(vel_box);
    mpc.SetTauBox(tau_box);
    mpc.SetForceBox(stance_force_box, swing_force_box);
    mpc.SetFrictionCone(std::move(friction_cone_constraint));
    mpc.SetSwingConstraint(std::move(swing_constraint));
    mpc.SetHolonomicConstraint(std::move(holonomic_constraint));
    mpc.SetCollisionConstraint(std::move(collision_constraint));
    mpc.SetPolytopeConstraint(std::move(polytope_constraint));
    std::cout << "===== MPC Constraints Added =====" << std::endl;

    mpc.SetVelTrackingCost(std::move(vel_tracking));
    mpc.SetTauTrackingCost(std::move(tau_tracking));
    mpc.SetForceTrackingCost(std::move(force_tracking));
    mpc.SetConfigTrackingCost(std::move(config_tracking));
    mpc.SetFowardKinematicsCost(std::move(fk_cost));

    std::cout << "===== MPC Costs Added =====" << std::endl;

    // Create an IC
    vectorx_t q = settings.q_target;
    // q(2) = 0.8;
    q(0) = 0;
    q(2) = 0.77;

    vectorx_t v = vectorx_t::Zero(g1.GetVelDim());
    // v(0) = 1;
    // v(1) = 2;
    // v(3) = 2;
    // v(4) = -1.5;
    // v(5) = 1;

    // Initial linearization
    Trajectory traj;
    SimpleTrajectory q_target(g1.GetConfigDim(), settings.nodes);
    SimpleTrajectory v_target(g1.GetVelDim(), settings.nodes);

    q_target.SetAllData(settings.q_target);
    v_target.SetAllData(settings.v_target);
    mpc.SetLinTrajConfig(q_target);
    mpc.SetLinTrajVel(v_target);

    for (int i = 0; i < settings.nodes; i++) {
        // auto lam = static_cast<double>(i)/static_cast<double>(settings.nodes);
        // std::cout << "lambda: " << lam << std::endl;
        // q_target.InsertData(i, (1-lam)*q + lam*settings.q_target);
        // v_target.InsertData(i, (1-lam)*v + lam*settings.v_target);

        // Sinusoidal z height
        // vectorx_t q_temp = settings.q_target;
        // q_temp(2) = 0.77 - 0.3*sin(0.1*i);
        // q_target[i] = q_temp;
        //
        // vectorx_t v_temp = settings.v_target;
        // v_temp(2) = -0.3*0.1*cos(0.1*i);
        // v_target[i] = v_temp;

        // std::cout << "[" << i << "] qz: " << q_temp(2) << ", vz: " << v_temp(2) << std::endl;
    }

    mpc.SetConfigTarget(q_target);
    mpc.SetVelTarget(v_target);

    // mpc.UpdateContactSchedule(cs);

    std::cout << "q ic: " << q.transpose() << std::endl;
    std::cout << "v ic: " << v.transpose() << std::endl;

    double time = 0;

    torc::utils::TORCTimer timer;
    timer.Tic();
    mpc.CreateQPData();
    mpc.Compute(time, q, v, traj);
    mpc.LogMPCCompute(time, q, v);
    timer.Toc();
    std::cout << "total compute time: " << timer.Duration<std::chrono::microseconds>().count()/1000.0 << "ms" << std::endl << std::endl;
    traj.ExportToCSV(std::filesystem::current_path() / "trajectory_output_1.csv");

    // timer.Tic();
    // mpc.Compute(q, v, traj);
    // timer.Toc();
    // std::cout << "total compute time: " << timer.Duration<std::chrono::microseconds>().count()/1000.0 << "ms" << std::endl << std::endl;
    //
    // timer.Tic();
    // mpc.Compute(q, v, traj);
    // timer.Toc();
    // std::cout << "total compute time: " << timer.Duration<std::chrono::microseconds>().count()/1000.0 << "ms" << std::endl << std::endl;

    // for (int i = 0; i < 250; i++) {
    //     mpc.Compute(q, v, traj);
    // }

    // double dt = -0.01;
    // cs.ShiftSwings(dt);
    // vectorx_t q_traj, v_traj;
    // q_traj = q;
    // v_traj = v;
    // // traj.GetConfigInterp(0.25, q_traj);
    // // traj.GetVelocityInterp(0.25, v_traj);
    // mpc.Compute(q_traj, v_traj, traj);
    //
    // cs.ShiftSwings(dt);
    // // traj.GetConfigInterp(0.25, q_traj);
    // // traj.GetVelocityInterp(0.25, v_traj);
    // mpc.Compute(q_traj, v_traj, traj);
    //
    // cs.ShiftSwings(dt);
    // mpc.Compute(q_traj, v_traj, traj);
    //
    // cs.ShiftSwings(dt);
    // mpc.Compute(q_traj, v_traj, traj);
    //
    // cs.ShiftSwings(dt);
    // mpc.Compute(q_traj, v_traj, traj);
    //
    // cs.ShiftSwings(dt);
    // mpc.Compute(q_traj, v_traj, traj);
    //
    // cs.ShiftSwings(dt);
    // mpc.Compute(q_traj, v_traj, traj);
    //
    // cs.ShiftSwings(dt);
    // mpc.Compute(q_traj, v_traj, traj);

    // mpc.Compute(q, v, traj);
    // mpc.Compute(q, v, traj);
    // mpc.Compute(q, v, traj);
    for (int i = 0; i < 50; i++) { // 50
        cs.ShiftSwings(-0.01);
        vectorx_t q_traj, v_traj;
        traj.GetConfigInterp(0.01, q_traj);
        traj.GetVelocityInterp(0.01, v_traj);
        mpc.UpdateContactSchedule(cs);
        time += 0.01;
        mpc.CreateQPData();

        // To try and make the problem slightly harder
        q_traj[0] += 0.005;
        q_traj[1] -= 0.005;
        q_traj[2] += 0.008;
        q_traj[8] += 0.005;
        q_traj[10] -= 0.009;

        v_traj[0] += 0.01;
        v_traj[1] -= 0.01;
        v_traj[2] += 0.005;

        mpc.Compute(time, q_traj, v_traj, traj);
        mpc.LogMPCCompute(time, q_traj, v_traj);
    }
    mpc.PrintNodeInfo();

    traj.ExportToCSV(std::filesystem::current_path() / "trajectory_output_2.csv");
}

int main() {
    std::thread thread(thread_function);
    thread.join();
    // pthread_t thread;
    // pthread_create(&thread, NULL, thread_function, NULL);
    // pthread_join(thread, NULL);
    return 0;
}

