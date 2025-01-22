//
// Created by zolkin on 1/18/25.
//

#include <torc_timer.h>

#include "DynamicsConstraint.h"
#include "hpipm_mpc.h"

int main() {
    // Create all the constraints
    using namespace torc::mpc;
    std::filesystem::path g1_urdf = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/g1_hand.urdf";

    std::filesystem::path mpc_config = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/g1_mpc_config.yaml";

    MpcSettings settings(mpc_config);
    settings.Print();

    const std::string pin_model_name = "test_pin_model";

    torc::models::FullOrderRigidBody g1("g1", g1_urdf, settings.joint_skip_names, settings.joint_skip_values);

    fs::path deriv_lib_path = fs::current_path();
    deriv_lib_path = deriv_lib_path / "deriv_libs";

    std::vector<std::string> contact_frames = settings.contact_frames;
    // --------------------------------- //
    // ---------- Constraints ---------- //
    // --------------------------------- //
    // ---------- Dynamics ---------- //
    std::vector<DynamicsConstraint> dynamics_constraints;
    // dynamics_constraints.emplace_back(DynamicsConstraint(g1, contact_frames, "g1_full_order",
    //     deriv_lib_path, false, true, 0, 5));
    // dynamics_constraints.emplace_back(g1, contact_frames, "g1_centroidal", deriv_lib_path,
    //     false, false, 5, settings.nodes);
    dynamics_constraints.emplace_back(g1, contact_frames, "g1_full_order",
        deriv_lib_path, settings.compile_derivs, true, 0, settings.nodes_full_dynamics);
    dynamics_constraints.emplace_back(g1, contact_frames, "g1_centroidal", deriv_lib_path,
        settings.compile_derivs, false, settings.nodes_full_dynamics, settings.nodes);

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

    // ---------- Friction Cone Constraints ---------- //
    FrictionConeConstraint friction_cone_constraint(0,settings.nodes, "friction_cone_cone",
        settings.friction_coef, settings.friction_margin, settings.deriv_lib_path, settings.compile_derivs);

    // ---------- Swing Constraints ---------- //
    SwingConstraint swing_constraint(2, settings.nodes, "swing_constraint", g1, contact_frames,
        settings.deriv_lib_path, settings.compile_derivs);

    // ---------- Holonomic Constraints ---------- //
    HolonomicConstraint holonomic_constraint(1, settings.nodes, "holonomic_constraint", g1, contact_frames,
        settings.deriv_lib_path, settings.compile_derivs);

    std::cout << "===== Constraints Created =====" << std::endl;

    // --------------------------------- //
    // ------------- Costs ------------- //
    // --------------------------------- //
    // ---------- Velocity Tracking ---------- //
    std::cout << settings.cost_data.at(1).weight.transpose() << std::endl;
    LinearLsCost vel_tracking(0, settings.nodes, "vel_tracking", settings.cost_data.at(1).weight,
        settings.deriv_lib_path, settings.compile_derivs);

    // ---------- Tau Tracking ---------- //
    LinearLsCost tau_tracking(0, settings.nodes, "tau_tracking", settings.cost_data.at(2).weight,
        settings.deriv_lib_path, settings.compile_derivs);

    // ---------- Force Tracking ---------- //
    LinearLsCost force_tracking(0, settings.nodes, "force_tracking", settings.cost_data.at(3).weight,
        settings.deriv_lib_path, settings.compile_derivs);

    // ---------- Config Tracking ---------- //
    ConfigTrackingCost config_tracking(0, settings.nodes, "config_tracking", settings.cost_data.at(0).weight,
        settings.deriv_lib_path, settings.compile_derivs, g1);

    // --------------------------------- //
    // ------- Contact Schedule -------- //
    // --------------------------------- //
    torc::mpc::ContactSchedule cs(settings.contact_frames);

    cs.InsertSwing("right_toe", 0.1, 0.4);
    cs.InsertSwing("right_heel", 0.1, 0.4);
    cs.InsertSwing("left_toe", 0.4, 0.7);
    cs.InsertSwing("left_heel", 0.4, 0.7);

    // --------------------------------- //
    // -------------- MPC -------------- //
    // --------------------------------- //
    HpipmMpc mpc(settings, g1);

    std::cout << "===== MPC Created =====" << std::endl;
    mpc.SetDynamicsConstraints(std::move(dynamics_constraints));
    mpc.SetConfigBox(config_box);
    mpc.SetVelBox(vel_box);
    mpc.SetTauBox(tau_box);
    mpc.SetFrictionCone(std::move(friction_cone_constraint));
    mpc.SetSwingConstraint(std::move(swing_constraint));
    mpc.SetHolonomicConstraint(std::move(holonomic_constraint));
    std::cout << "===== MPC Constraints Added =====" << std::endl;

    mpc.SetVelTrackingCost(std::move(vel_tracking));
    mpc.SetTauTrackingCost(std::move(tau_tracking));
    mpc.SetForceTrackingCost(std::move(force_tracking));
    mpc.SetConfigTrackingCost(std::move(config_tracking));

    std::cout << "===== MPC Costs Added =====" << std::endl;

    Trajectory traj;
    SimpleTrajectory q_target(g1.GetConfigDim(), settings.nodes);
    q_target.SetAllData(settings.q_target);
    mpc.SetConfigTarget(q_target);

    SimpleTrajectory v_target(g1.GetVelDim(), settings.nodes);
    v_target.SetAllData(settings.v_target);
    mpc.SetVelTarget(v_target);

    mpc.SetLinTrajConfig(q_target);
    mpc.SetLinTrajVel(v_target);

    mpc.UpdateContactSchedule(cs);

    // Create an IC
    vectorx_t q = settings.q_target;
    // q(2) = 0.8;
    q(0) = 1;

    torc::utils::TORCTimer timer;
    timer.Tic();
    mpc.Compute(q, vectorx_t::Zero(g1.GetVelDim()), traj);
    timer.Toc();
    std::cout << "total compute time: " << timer.Duration<std::chrono::microseconds>().count()/1000.0 << "ms" << std::endl << std::endl;

    mpc.GetConstraintViolation();

    timer.Tic();
    mpc.Compute(q, vectorx_t::Zero(g1.GetVelDim()), traj);
    timer.Toc();
    std::cout << "total compute time: " << timer.Duration<std::chrono::microseconds>().count()/1000.0 << "ms" << std::endl << std::endl;

    timer.Tic();
    mpc.Compute(q, vectorx_t::Zero(g1.GetVelDim()), traj);
    timer.Toc();
    std::cout << "total compute time: " << timer.Duration<std::chrono::microseconds>().count()/1000.0 << "ms" << std::endl << std::endl;

    traj.ExportToCSV(std::filesystem::current_path() / "trajectory_output.csv");

}

