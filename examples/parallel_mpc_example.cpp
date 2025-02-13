//
// Created by zolkin on 2/12/25.
//

#include <iostream>

#include <pthread.h>
#include <omp.h>

#include "hpipm_mpc.h"

std::pair<double, double> ComputeStats(const std::vector<double>& times) {
    double mean = 0;
    for (const auto& t : times) {
        mean += t;
    }
    mean /= times.size();

    double std_dev = 0;
    for (const auto& t : times) {
        std_dev += (t - mean) * (t - mean);
    }

    std_dev = sqrt(std_dev / times.size());

    return {mean, std_dev};
}

std::pair<double, double> SetupAndComputeMPC(int num_solves, torc::mpc::HpipmMpc& mpc,
    torc::mpc::ContactSchedule& cs, const torc::mpc::vectorx_t& q, const torc::mpc::vectorx_t& v) {

    // std::cout << "q ic: " << q.transpose() << std::endl;
    // std::cout << "v ic: " << v.transpose() << std::endl;

    double time = 0;
    torc::mpc::Trajectory traj;

    std::vector<double> solve_time;

    for (int i = 0; i < num_solves; i++) {
        torc::utils::TORCTimer timer;
        timer.Tic();
        if (i != 0) {
            cs.ShiftSwings(-0.01);
            torc::mpc::vectorx_t q_traj, v_traj;
            traj.GetConfigInterp(0.01, q_traj);
            traj.GetVelocityInterp(0.01, v_traj);
            mpc.UpdateContactSchedule(cs);
            time += 0.01;

            mpc.CreateQPData();

            // // To try and make the problem slightly harder
            // q_traj[0] += 0.005;
            // q_traj[1] -= 0.005;
            // q_traj[2] += 0.008;
            // q_traj[8] += 0.005;
            // q_traj[10] -= 0.009;
            //
            // v_traj[0] += 0.01;
            // v_traj[1] -= 0.01;
            // v_traj[2] += 0.005;

            mpc.Compute(time, q_traj, v_traj, traj);
            mpc.LogMPCCompute(time, q_traj, v_traj);
        } else {
            mpc.CreateQPData();
            mpc.Compute(time, q, v, traj);
            mpc.LogMPCCompute(time, q, v);
        }
        timer.Toc();
        solve_time.push_back(timer.Duration<std::chrono::microseconds>().count()/1000.0);
    }

    // Compute mean and standard deviation for the times

    return ComputeStats(solve_time);
}

void thread_compute(int num_solves, torc::mpc::HpipmMpc& mpc,
    torc::mpc::ContactSchedule& cs, const torc::mpc::vectorx_t& q, const torc::mpc::vectorx_t& v) {
    // Pin to a CPU
    // Prevents uncessary cache misses
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(2, &cpuset);  // Pin to CPU 2

    pthread_t thread = pthread_self();
    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_setaffinity_np");
    }

    const auto [single_mean, single_std_dev] = SetupAndComputeMPC(num_solves, mpc, cs, q, v);
    std::cout << "===== NUM SOLVES: " << num_solves << " ====" << std::endl;
    std::cout << "Single Thread mean time: " << single_mean << std::endl;
    std::cout << "Single Thread std dev time: " << single_std_dev << std::endl;
}

int main() {
    using namespace torc::mpc;

    constexpr int NUM_THREADS = 6; //5;
    omp_set_num_threads(NUM_THREADS);

    // TODO: Make the MPC more thread safe
    std::vector<torc::mpc::HpipmMpc> mpcs;
    std::vector<torc::mpc::ContactSchedule> cs;
    torc::mpc::vectorx_t q, v;
    for (int i = 0; i < NUM_THREADS + 1; i++) {
        // Create all the constraints
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
        cs.emplace_back(settings.contact_frames);

        cs.back().InsertSwing("right_toe", 0.1, 0.5);
        cs.back().InsertSwing("right_heel", 0.1, 0.5);
        cs.back().InsertSwing("left_toe", 0.5, 0.9);
        cs.back().InsertSwing("left_heel", 0.5, 0.9);

        cs.back().InsertSwing("right_toe", 0.9, 1.2);
        cs.back().InsertSwing("right_heel", 0.9, 1.2);
        cs.back().InsertSwing("left_toe", 1.2, 1.6);
        cs.back().InsertSwing("left_heel", 1.2, 1.6);

        cs.back().InsertSwing("right_toe", 1.6, 2.0);
        cs.back().InsertSwing("right_heel", 1.6, 2.0);
        cs.back().InsertSwing("left_toe", 2.0, 2.4);
        cs.back().InsertSwing("left_heel", 2.0, 2.4);

        cs.back().InsertSwing("right_toe", 2.4, 2.8);
        cs.back().InsertSwing("right_heel", 2.4, 2.8);
        cs.back().InsertSwing("left_toe", 2.8, 3.2);
        cs.back().InsertSwing("left_heel", 2.8, 3.2);
        // --------------------------------- //
        // -------------- MPC -------------- //
        // --------------------------------- //
        HpipmMpc mpc(settings, g1);
        mpcs.push_back(std::move(mpc));

        // MpcSettings settings2 = settings;
        // mpcs.emplace_back(std::move(settings2), g1);

        std::cout << "===== MPC Created =====" << std::endl;
        mpcs.back().SetDynamicsConstraints(std::move(dynamics_constraints));
        mpcs.back().SetCentroidalDynamicsConstraints(std::move(centroidal_dynamics));
        // mpc.SetSrbConstraint(std::move(srb_dynamics));
        mpcs.back().SetConfigBox(config_box);
        mpcs.back().SetVelBox(vel_box);
        mpcs.back().SetTauBox(tau_box);
        mpcs.back().SetForceBox(stance_force_box, swing_force_box);
        mpcs.back().SetFrictionCone(std::move(friction_cone_constraint));
        mpcs.back().SetSwingConstraint(std::move(swing_constraint));
        mpcs.back().SetHolonomicConstraint(std::move(holonomic_constraint));
        mpcs.back().SetCollisionConstraint(std::move(collision_constraint));
        mpcs.back().SetPolytopeConstraint(std::move(polytope_constraint));
        std::cout << "===== MPC Constraints Added =====" << std::endl;

        mpcs.back().SetVelTrackingCost(std::move(vel_tracking));
        mpcs.back().SetTauTrackingCost(std::move(tau_tracking));
        mpcs.back().SetForceTrackingCost(std::move(force_tracking));
        mpcs.back().SetConfigTrackingCost(std::move(config_tracking));
        mpcs.back().SetFowardKinematicsCost(std::move(fk_cost));

        std::cout << "===== MPC Costs Added =====" << std::endl;
        // ---------------------------------------------------- //
        // ---------------------------------------------------- //
        // ---------------------------------------------------- //

        // Create an IC
        q = settings.q_target;
        q(0) = 0;
        q(2) = 0.77;

        v = vectorx_t::Zero(g1.GetVelDim());

        // Initial linearization
        SimpleTrajectory q_target(g1.GetConfigDim(), settings.nodes);
        SimpleTrajectory v_target(g1.GetVelDim(), settings.nodes);

        q_target.SetAllData(settings.q_target);
        v_target.SetAllData(settings.v_target);
        mpcs.back().SetLinTrajConfig(q_target);
        mpcs.back().SetLinTrajVel(v_target);

        mpcs.back().SetConfigTarget(q_target);
        mpcs.back().SetVelTarget(v_target);

        mpcs.back().UpdateContactSchedule(cs.back());
    }

    const int num_solves = 100;

    // std::thread single_solve_thread(thread_compute, num_solves, mpcs[0], cs[0], q, v);
    // single_solve_thread.join();

    // // Pin to a CPU
    // // Prevents uncessary cache misses
    // cpu_set_t cpuset;
    // CPU_ZERO(&cpuset);
    // CPU_SET(2, &cpuset);  // Pin to CPU 2
    //
    // pthread_t thread = pthread_self();
    // if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset) != 0) {
    //     perror("pthread_setaffinity_np");
    // }


    const auto [single_mean, single_std_dev] = SetupAndComputeMPC(num_solves, mpcs[0], cs[0], q, v);
    std::cout << "===== NUM SOLVES: " << num_solves << " ====" << std::endl;
    std::cout << "Single Thread mean time: " << single_mean << std::endl;
    std::cout << "Single Thread std dev time: " << single_std_dev << std::endl;

    // Beginning of parallel region
    #pragma omp parallel
    {
        const auto [single_mean, single_std_dev] = SetupAndComputeMPC(num_solves, mpcs.at(omp_get_thread_num() + 1),
            cs.at(omp_get_thread_num() + 1), q, v);
        std::cout << "===== NUM SOLVES: " << num_solves << " ====" << std::endl;
        std::cout << "Thread " << omp_get_thread_num() << " mean time: " << single_mean << std::endl;
        std::cout << "Thread " << omp_get_thread_num() << " std dev time: " << single_std_dev << std::endl;
        // printf("Hello World... from thread = %d\n",
        // omp_get_thread_num());
    }

}
