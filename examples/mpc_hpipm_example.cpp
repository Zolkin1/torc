//
// Created by zolkin on 1/18/25.
//

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
    std::cout << "g1 config dim: " << g1.GetConfigDim() << std::endl;
    std::cout << "g1 vel dim: " << g1.GetVelDim() << std::endl;

    fs::path deriv_lib_path = fs::current_path();
    deriv_lib_path = deriv_lib_path / "deriv_libs";

    std::vector<std::string> contact_frames = {"left_toe", "left_heel", "right_toe", "right_heel"};

    // ---------- Dynamics ---------- //
    std::vector<DynamicsConstraint> dynamics_constraints;
    // dynamics_constraints.emplace_back(DynamicsConstraint(g1, contact_frames, "g1_full_order",
    //     deriv_lib_path, false, true, 0, 5));
    // dynamics_constraints.emplace_back(g1, contact_frames, "g1_centroidal", deriv_lib_path,
    //     false, false, 5, settings.nodes);
    dynamics_constraints.emplace_back(g1, contact_frames, "g1_full_order",
        deriv_lib_path, settings.compile_derivs, true, 0, 5);
    dynamics_constraints.emplace_back(g1, contact_frames, "g1_centroidal", deriv_lib_path,
        settings.compile_derivs, false, 5, settings.nodes);

    // ---------- Box Constraints ---------- //
    // Config
    std::vector<int> config_lims_idxs;
    for (int i = 0; i < g1.GetConfigDim() - torc::mpc::FLOATING_BASE; ++i) {
        config_lims_idxs.push_back(i + FLOATING_BASE);
    }
    BoxConstraint config_box(1, settings.nodes, "config_box",
        g1.GetLowerConfigLimits().tail(g1.GetConfigDim() - torc::mpc::FLOATING_BASE),
        g1.GetUpperConfigLimits().tail(g1.GetConfigDim() - torc::mpc::FLOATING_BASE),
        config_lims_idxs);

    // Vel
    std::vector<int> vel_lims_idxs;
    for (int i = 0; i < g1.GetVelDim() - torc::mpc::FLOATING_VEL; ++i) {
        vel_lims_idxs.push_back(i + FLOATING_BASE);
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
    BoxConstraint tau_box(1, settings.nodes, "tau_box",
        -g1.GetTorqueJointLimits().tail(g1.GetVelDim() - torc::mpc::FLOATING_VEL),
        g1.GetTorqueJointLimits().tail(g1.GetVelDim() - torc::mpc::FLOATING_VEL),
        tau_lims_idxs);

    // ---------- Friction Cone Constraints ---------- //
    FrictionConeConstraint friction_cone_constraint(0, settings.nodes, "friction_cone_cone",
        settings.friction_coef, settings.friction_margin, settings.deriv_lib_path, settings.compile_derivs);

    // ---------- Swing Constraints ---------- //
    SwingConstraint swing_constraint(1, settings.nodes, "swing_constraint", g1, contact_frames,
        settings.deriv_lib_path, settings.compile_derivs);

    // ---------- Holonomic Constraints ---------- //
    HolonomicConstraint holonomic_constraint(1, settings.nodes, "holonomic_constraint", g1, contact_frames,
        settings.deriv_lib_path, settings.compile_derivs);

    std::cout << "===== Constraints Created =====" << std::endl;

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
    mpc.CreateConstraints();
    mpc.CreateCost();
    mpc.Compute(g1.GetNeutralConfig(), vectorx_t::Zero(g1.GetVelDim()));
}

