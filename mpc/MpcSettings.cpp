//
// Created by zolkin on 1/18/25.
//

#include <iostream>
#include "yaml-cpp/yaml.h"

#include "MpcSettings.h"

namespace torc::mpc {
    MpcSettings::MpcSettings(const fs::path& config_file) {
        ParseConfigFile(config_file);
    }

    void MpcSettings::ParseConfigFile(const fs::path& config_file) {
        config_file_ = config_file;

        ParseGeneralSettings();
        ParseJointDefaults();
        ParseSolverSettings();
        ParseConstraintSettings();
        ParseCostSettings();
        ParseLineSearchSettings();
        ParseContactSettings();
    }

    void MpcSettings::ParseJointDefaults() {
        // Read in the yaml file.
        YAML::Node config;
        try {
            config = YAML::LoadFile(config_file_);
        } catch (...) {
            throw std::runtime_error("Could not load the configuration file!");
        }

        // ---------- Joint Default Settings ---------- //
        if (config["joint_defaults"]) {
            std::cout << "Reading in joint defaults." << std::endl;
            YAML::Node joint_defualt_settings = config["joint_defaults"];
            joint_skip_names = joint_defualt_settings["joints"].as<std::vector<std::string>>();
            joint_skip_values = joint_defualt_settings["values"].as<std::vector<double>>();
        }

        if (joint_skip_names.size() != joint_skip_values.size()) {
            throw std::runtime_error("Joint names and joint values in joint_defaults do not match!");
        }
    }

    void MpcSettings::ParseGeneralSettings() {
        YAML::Node config;
        try {
            config = YAML::LoadFile(config_file_);
        } catch (...) {
            throw std::runtime_error("Could not load the configuration file!");
        }

        // ---------- General Settings ---------- //
        if (!config["general_settings"]) {
            throw std::runtime_error("No general_settings provided!");
        } else {
            YAML::Node general_settings = config["general_settings"];
            if (general_settings["nodes"]) {
                nodes = general_settings["nodes"].as<int>();
                if (nodes < 3) {
                    throw std::invalid_argument("nodes must be >= 3!");
                }
            } else {
                throw std::runtime_error("Number of nodes not specified!");
            }

            if (general_settings["verbose"]) {
                verbose = general_settings["verbose"].as<bool>();
            }

            int node_type = 0;
            if (general_settings["node_dt_type"]) {
                if (general_settings["node_dt_type"].as<std::string>() == "two_groups") {
                    node_type = 1;
                } else if (general_settings["node_dt_type"].as<std::string>() == "Adaptive") {
                    throw std::runtime_error("Adaptive node dt not implemented yet!");
                } else if (general_settings["node_dt_type"].as<std::string>() == "Even") {
                    node_type = 0;
                } else {
                    throw std::runtime_error("Node dt type not supported!");
                }
            }
            if (node_type == 0) {
                if (general_settings["node_dt"]) {
                    const auto dt_in = general_settings["node_dt"].as<double>();
                    dt.resize(nodes);
                    for (double & it : dt) {
                        it = dt_in;
                    }
                } else {
                    throw std::runtime_error("Node dt not specified!");
                }
            } else if (node_type == 1) {
                if (general_settings["node_group_1_n"] && general_settings["node_group_2_n"]
                        && general_settings["node_dt_1"] && general_settings["node_dt_2"]) {
                    dt.resize(nodes);

                    if (general_settings["node_group_1_n"].as<int>() + general_settings["node_group_2_n"].as<int>() != nodes) {
                        throw std::runtime_error("Node groups don't sum to the nodes!");
                    }

                    for (int i = 0; i < general_settings["node_group_1_n"].as<int>(); i++) {
                        dt[i] = general_settings["node_dt_1"].as<double>();
                    }

                    for (int i = general_settings["node_group_1_n"].as<int>(); i < general_settings["node_group_2_n"].as<int>() + general_settings["node_group_1_n"].as<int>(); i++) {
                        dt[i] = general_settings["node_dt_2"].as<double>();
                    }
                } else {
                    throw std::runtime_error("Node group 1 or 2 n not specified or the dt's are not specified!");
                }
            }

            if (general_settings["compile_derivatives"]) {
                compile_derivs = general_settings["compile_derivatives"].as<bool>();
            }

            if (general_settings["deriv_lib_path"]) {
                deriv_lib_path = general_settings["deriv_lib_path"].as<std::string>();
            } else {
                deriv_lib_path = fs::current_path();
                deriv_lib_path = deriv_lib_path / "deriv_libs";
            }

            if (general_settings["base_frame"]) {
                base_frame = general_settings["base_frame"].as<std::string>();
            } else {
                throw std::runtime_error("No base frame name provided in configuration file!");
            }

            if (general_settings["scale_cost"]) {
                scale_cost = general_settings["scale_cost"].as<bool>();
            } else {
                scale_cost = false;
            }

            if (general_settings["max_initial_solves"]) {
                max_initial_solves = general_settings["max_initial_solves"].as<int>();
            } else {
                max_initial_solves = 10;
            }

            if (general_settings["initial_constraint_tol"]) {
                initial_solve_tolerance = general_settings["initial_constraint_tol"].as<double>();
            } else {
                initial_solve_tolerance = 1e-2;
            }

            if (general_settings["nodes_full_dynamics"]) {
                nodes_full_dynamics = general_settings["nodes_full_dynamics"].as<int>();
                if (nodes_full_dynamics > nodes - 1) {
                    throw std::invalid_argument("The nodes with full dynamics must be <= total nodes - 1");
                }
            } else {
                nodes_full_dynamics = std::min(5, nodes - 1);
            }

            terminal_weight = (general_settings["terminal_cost_weight"] ? general_settings["terminal_cost_weight"].as<double>() : 1.0);
        }
    }

    void MpcSettings::ParseSolverSettings() {
        YAML::Node config;
        try {
            config = YAML::LoadFile(config_file_);
        } catch (...) {
            throw std::runtime_error("Could not load the configuration file!");
        }

        if (!config["solver_settings"]) {
            if (verbose) {
                std::cout << "[MPC] No solver settings given. Using defaults." << std::endl;
            }
        } else {
            YAML::Node solver_settings = config["solver_settings"];
            if (solver_settings["alpha_min"]) {
                qp_settings.alpha_min = solver_settings["alpha_min"].as<double>();
            }
            if (solver_settings["max_iter"]) {
                qp_settings.iter_max = solver_settings["max_iter"].as<int>();
            }
            if (solver_settings["mu0"]) {
                qp_settings.mu0 = solver_settings["mu0"].as<double>();
            }
            if (solver_settings["tol_stat"]) {
                qp_settings.tol_stat = solver_settings["tol_stat"].as<double>();
            }
            if (solver_settings["tol_eq"]) {
                qp_settings.tol_eq = solver_settings["tol_eq"].as<double>();
            }
            if (solver_settings["tol_ineq"]) {
                qp_settings.tol_ineq = solver_settings["tol_ineq"].as<double>();
            }
            if (solver_settings["tol_comp"]) {
                qp_settings.tol_comp = solver_settings["tol_comp"].as<double>();
            }
            if (solver_settings["reg_prim"]) {
                qp_settings.reg_prim = solver_settings["reg_prim"].as<double>();
            }
            if (solver_settings["warm_start"]) {
                qp_settings.warm_start = solver_settings["warm_start"].as<int>();
                if (qp_settings.warm_start != 0 && qp_settings.warm_start != 1) {
                    throw std::invalid_argument("[MpcSettings] The warm_start parameter must be 0 or 1!");
                }
            }
            if (solver_settings["pred_corr"]) {
                qp_settings.pred_corr = solver_settings["pred_corr"].as<int>();
                if (qp_settings.pred_corr != 0 && qp_settings.pred_corr != 1) {
                    throw std::invalid_argument("[MpcSettings] The pred_corr parameter must be 0 or 1!");
                }
            }
            if (solver_settings["ric_alg"]) {
                qp_settings.ric_alg = solver_settings["ric_alg"].as<int>();
                if (qp_settings.ric_alg != 0 && qp_settings.ric_alg != 1) {
                    throw std::invalid_argument("[MpcSettings] The ric_alg parameter must be 0 or 1!");
                }
            }
            if (solver_settings["split_step"]) {
                qp_settings.split_step = solver_settings["split_step"].as<int>();
                if (qp_settings.split_step != 0 && qp_settings.split_step != 1) {
                    throw std::invalid_argument("[MpcSettings] The split_step parameter must be 0 or 1!");
                }
            }
            if (solver_settings["mode"]) {
                if (solver_settings["mode"].as<std::string>() == "Balance") {
                    qp_settings.mode = hpipm::HpipmMode::Balance;
                } else if (solver_settings["mode"].as<std::string>() == "SpeedAbs") {
                    qp_settings.mode = hpipm::HpipmMode::SpeedAbs;
                } else if (solver_settings["mode"].as<std::string>() == "Speed") {
                    qp_settings.mode = hpipm::HpipmMode::Speed;
                } else if (solver_settings["mode"].as<std::string>() == "Robust") {
                    qp_settings.mode = hpipm::HpipmMode::Robust;
                } else {
                    throw std::runtime_error("[MpcSettings] HPIPM mode must be one of: {Balance, SpeedAbs, Speed, Robust}");
                }
            }
        }
    }

    void MpcSettings::ParseConstraintSettings() {
        YAML::Node config;
        try {
            config = YAML::LoadFile(config_file_);
        } catch (...) {
            throw std::runtime_error("Could not load the configuration file!");
        }

        if (!config["constraints"]) {
            throw std::runtime_error("No constraint settings provided!");
        }
        YAML::Node constraint_settings = config["constraints"];
        friction_coef = constraint_settings["friction_coef"].as<double>();
        max_grf = constraint_settings["max_grf"].as<double>();
        friction_margin = (constraint_settings["friction_margin"] ? constraint_settings["friction_margin"].as<double>() : 0.05);
        polytope_delta = (constraint_settings["polytope_delta"] ? constraint_settings["polytope_delta"].as<double>() : 0.0);
        polytope_shrinking_rad = (constraint_settings["polytope_shrinking_rad"]
            ? constraint_settings["polytope_shrinking_rad"].as<double>() : 0.4);

        YAML::Node collision_settings = constraint_settings["collisions"];
        if (!collision_settings.IsSequence()) {
            throw std::runtime_error("Collision settings must be a sequence!");
        }

        for (const auto& collision : collision_settings) {
            const auto frame1 = collision["frame1"].as<std::string>();
            const auto frame2 = collision["frame2"].as<std::string>();
            collision_frames.emplace_back(frame1, frame2);

            const auto r1 = collision["radius1"].as<double>();
            const auto r2 = collision["radius2"].as<double>();
            collision_radii.emplace_back(r1, r2);
        }
    }

    void MpcSettings::ParseCostSettings() {
        YAML::Node config;
        try {
            config = YAML::LoadFile(config_file_);
        } catch (...) {
            throw std::runtime_error("Could not load the configuration file!");
        }

        if (!config["costs"]) {
            throw std::runtime_error("No cost settings provided!");
        }
        YAML::Node cost_settings = config["costs"];

        if (!cost_settings.IsSequence()) {
            throw std::runtime_error("costs must be a sequence!");
        }

        cost_data.resize(cost_settings.size());
        int idx = 0;
        for (const auto& cost_term : cost_settings) {
            if (cost_term["type"]) {
                const std::string type = cost_term["type"].as<std::string>();
                if (type == "ConfigurationTracking") {
                    cost_data[idx].type = CostTypes::Configuration;
                } else if (type == "VelocityTracking") {
                    cost_data[idx].type = CostTypes::VelocityTracking;
                } else if (type == "TorqueRegularization") {
                    cost_data[idx].type = CostTypes::TorqueReg;
                } else if (type == "ForceRegularization") {
                    cost_data[idx].type = CostTypes::ForceReg;
                } else if (type == "ForwardKinematics") {
                    cost_data[idx].type = CostTypes::ForwardKinematics;
                } else {
                    throw std::runtime_error("Provided cost type does not exist!");
                }
            } else {
                throw std::runtime_error("Cost term must include a type!");
            }

            if (cost_term["name"]) {
                cost_data[idx].constraint_name = cost_term["name"].as<std::string>();
            } else {
                throw std::runtime_error("Cost term must include a name!");
            }

            if (cost_term["weight"]) {
                std::vector<double> weight = cost_term["weight"].as<std::vector<double>>();
                cost_data[idx].weight = utils::StdToEigenVector(weight);
            } else {
                throw std::runtime_error("Cost term must include a weight!");
            }

            if (cost_term["frame"]) {
                cost_data[idx].frame_name = cost_term["frame"].as<std::string>();
            }

            if (cost_term["mu"]) {
                cost_data[idx].mu = cost_term["mu"].as<double>();
            }
            if (cost_term["delta"]) {
                cost_data[idx].delta = cost_term["delta"].as<double>();
            }

            idx++;
        }

        if (cost_data.size() == 0) {
            throw std::runtime_error("No cost terms chosen!");
        }
    }

    void MpcSettings::ParseContactSettings() {
        YAML::Node config;
        try {
            config = YAML::LoadFile(config_file_);
        } catch (...) {
            throw std::runtime_error("Could not load the configuration file!");
        }

        if (!config["contacts"]) {
            throw std::runtime_error("No contact settings provided!");
        }
        YAML::Node contact_settings = config["contacts"];
        contact_frames = contact_settings["contact_frames"].as<std::vector<std::string>>();
        num_contact_locations = contact_frames.size();

        if (contact_settings["hip_offsets"]) {
            hip_offsets = contact_settings["hip_offsets"].as<std::vector<double>>();
            if (hip_offsets.size() != 2*contact_frames.size()) {
                throw std::runtime_error("Invalid number of hip offsets provided! Must match the number of contacts x 2!");
            }
        }

        apex_height = contact_settings["apex_height"].as<double>();
        apex_time = contact_settings["apex_time"].as<double>();
        default_ground_height = contact_settings["default_ground_height"].as<double>();

        if (apex_height < 0) {
            throw std::runtime_error("Invalid apex height provided!");
        }
        if (apex_time < 0 || apex_time > 1) {
            throw std::runtime_error("Invalid apex time provided!");
        }
    }

    void MpcSettings::ParseLineSearchSettings() {
        YAML::Node config;
        try {
            config = YAML::LoadFile(config_file_);
        } catch (...) {
            throw std::runtime_error("Could not load the configuration file!");
        }

        if (!config["line_search"]) {
            throw std::runtime_error("No line search setting provided!");
        } else {
            YAML::Node ls_settings = config["line_search"];
            ls_eta = (ls_settings["armijo_constant"] ? ls_settings["armijo_constant"].as<double>() : 1e-4);
            ls_alpha_min = (ls_settings["alpha_min"] ? ls_settings["alpha_min"].as<double>() : 1e-4);
            ls_theta_max = (ls_settings["large_constraint_vio"] ? ls_settings["large_constraint_vio"].as<double>() : 1e-2);
            ls_theta_min = (ls_settings["small_constraint_vio"] ? ls_settings["small_constraint_vio"].as<double>() : 1e-6);
            ls_gamma_theta = (ls_settings["constraint_reduction_mult"] ? ls_settings["constraint_reduction_mult"].as<double>() : 1e-6);
            ls_gamma_alpha = (ls_settings["alpha_step"] ? ls_settings["alpha_step"].as<double>() : 0.5);
            ls_gamma_phi = (ls_settings["cost_reduction_mult"] ? ls_settings["cost_reduction_mult"].as<double>() : 1e-6);
        }
    }


    void MpcSettings::Print() {
        using std::setw;
            using std::setfill;

            const int total_width = 50;


            auto time_now = std::chrono::system_clock::now();
            std::time_t time1_now = std::chrono::system_clock::to_time_t(time_now);

            std::cout << setfill('=') << setw(total_width/2 - 7) << "" << " MPC Settings " << setw(total_width/2 - 7) << "" << std::endl;
            std::cout << "Current time: " << std::ctime(&time1_now);

            std::cout << "General settings: " << std::endl;
            std::cout << "\tVerbose: " << (verbose ? "True" : "False") << std::endl;
            std::cout << "\tNodes: " << nodes << std::endl;
            std::cout << "\tCompile derivatives: " << (compile_derivs ? "True" : "False") << std::endl;
            std::cout << "\tDerivative library location: " << deriv_lib_path.string() << std::endl;
            std::cout << "\tScale cost: " << (scale_cost ? "True" : "False") << std::endl;
            std::cout << "\tMax initial solves: " << max_initial_solves << std::endl;
            std::cout << "\tInitial constraint tolerance: " << initial_solve_tolerance << std::endl;
            std::cout << "\tNodes with full dynamics: " << nodes_full_dynamics << std::endl;

            std::cout << "Solver settings: " << std::endl;
            // std::cout << "\tRelative tolerance: " << osqp_settings_.eps_rel << std::endl;
            // std::cout << "\tAbsolute tolerance: " << osqp_settings_.eps_abs << std::endl;
            // std::cout << "\tVerbose: " << (osqp_settings_.verbose ? "True" : "False") << std::endl;
            // std::cout << "\tPolish: " << (osqp_settings_.polish ? "True" : "False") << std::endl;
            // std::cout << "\trho: " << osqp_settings_.rho << std::endl;
            // std::cout << "\talpha: " << osqp_settings_.alpha << std::endl;
            // std::cout << "\tsigma: " << osqp_settings_.sigma << std::endl;
            // std::cout << "\tAdaptive rho: " << (osqp_settings_.adaptive_rho ? "True" : "False") << std::endl;
            // std::cout << "\tMax iterations: " << osqp_settings_.max_iter << std::endl;
            // std::cout << "\tScaling: " << osqp_settings_.scaling << std::endl;

            std::cout << "Constraints:" << std::endl;
            std::cout << "\tFriction coefficient: " << friction_coef << std::endl;
            std::cout << "\tFriction margin: " << friction_margin << std::endl;
            std::cout << "\tMaximum ground reaction force: " << max_grf << std::endl;
            std::cout << "\tPolytope delta: " << polytope_delta << std::endl;
            std::cout << "\tPolytope shrinking rad: " << polytope_shrinking_rad << std::endl;

            std::cout << "Costs:" << std::endl;
            for (const auto& data : cost_data) {
                std::cout << "\tCost name: " << data.constraint_name << std::endl;
                std::cout << "\t\tWeight: " << data.weight.transpose() << std::endl;
                if (data.frame_name != "") {
                    std::cout << "\t\tFrame name: " << data.frame_name << std::endl;
                }
            }
            std::cout << "\tTerminal cost weight: " << terminal_weight << std::endl;

            std::cout << "Contacts:" << std::endl;
            std::cout << "\tApex height: " << apex_height << std::endl;
            std::cout << "\tApex time: " << apex_time << std::endl;
            std::cout << "\tDefault ground height: " << default_ground_height << std::endl;
            std::cout << "\tNumber of contact locations: " << num_contact_locations << std::endl;
            std::cout << "\tContact frames: [ ";
            for (const auto& frame : contact_frames) {
                std::cout << frame << " ";
            }
            std::cout << "]" << std::endl;

            std::cout << "Line search:" << std::endl;
            std::cout << "\tArmijo constant: " << ls_eta << std::endl;
            std::cout << "\tLarge constraint violation: " << ls_theta_max << std::endl;
            std::cout << "\tSmall constraint violation: " << ls_theta_min << std::endl;
            std::cout << "\tConstraint reduction multiplier: " << ls_gamma_theta << std::endl;
            std::cout << "\tCost reduction multiplier: " << ls_gamma_phi << std::endl;
            std::cout << "\tLine search step: " << ls_gamma_alpha << std::endl;
            std::cout << "\tSmallest step: " << ls_alpha_min << std::endl;

            // std::cout << "Size: " << std::endl;
            // std::cout << "\tDecision variables: " << GetNumDecisionVars() << std::endl;
            // std::cout << "\tConstraints: " << GetNumConstraints() << std::endl;

            // std::cout << "Robot: " << std::endl;
            // std::cout << "\tName: " << robot_model_->GetUrdfRobotName() << std::endl;
            // std::cout << "\tDoFs: " << config_dim_ - FLOATING_BASE << std::endl;
            // std::cout << "\tUpper Configuration Bounds: " << robot_model_->GetUpperConfigLimits().transpose() << std::endl;
            // std::cout << "\tLower Configuration Bounds: " << robot_model_->GetLowerConfigLimits().transpose() << std::endl;
            // std::cout << "\tVelocity Bounds: " << robot_model_->GetVelocityJointLimits().transpose() << std::endl;
            // std::cout << "\tTorque Bounds: " << robot_model_->GetTorqueJointLimits().transpose() << std::endl;

            std::cout << setfill('=') << setw(total_width) << "" << std::endl;

    }



}