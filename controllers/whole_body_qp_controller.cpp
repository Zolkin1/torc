//
// Created by zolkin on 7/17/24.
//

#include "whole_body_qp_controller.h"
#include "yaml-cpp/yaml.h"

namespace torc::controllers {
    WholeBodyQPController::WholeBodyQPController(const std::string& name)
        : name_(name) {}

    WholeBodyQPController::WholeBodyQPController(const std::string& name, const models::FullOrderRigidBody& model)
        : WholeBodyQPController(name) {
        model_ = std::make_unique<models::FullOrderRigidBody>(model);
    }

    WholeBodyQPController::WholeBodyQPController(const std::string& name, const std::filesystem::path& urdf)
        : WholeBodyQPController(name) {
        std::string model_name = name_ + "_model";
        model_ = std::make_unique<models::FullOrderRigidBody>(model_name, urdf);
    }

    WholeBodyQPController::WholeBodyQPController(const std::string& name, const std::filesystem::path& urdf,
                                                 const fs::path& config_file_path)
        : WholeBodyQPController(name, urdf) {
        config_file_path_ = config_file_path;
        ParseUpdateSettings();
    }

    void WholeBodyQPController::UpdateConfigFile(const fs::path& config_file_path) {
        config_file_path_ = config_file_path;
        ParseUpdateSettings();
    }

    void WholeBodyQPController::ParseUpdateSettings() {
        // Read in the file from the config file
        if (!fs::exists(config_file_path_)) {
            throw std::runtime_error("[WBC] Invalid configuration file path!");
        }

        // Parse the yaml
        YAML::Node config;
        try {
            config = YAML::LoadFile(config_file_path_);
        } catch (...) {
            throw std::runtime_error("Could not load the configuration file!");
        }

        if (!config["solver_settings"]) {
            std::cout << "[WBC Controller] No solver settings given. Using defaults." << std::endl;
        } else {
            YAML::Node solver_settings = config["solver_settings"];
            solvers::OSQPInterfaceSettings qp_settings_;
            qp_settings_.rel_tol = (solver_settings["rel_tol"]) ? solver_settings["rel_tol"].as<double>() : -1.0;
            qp_settings_.abs_tol = (solver_settings["abs_tol"]) ? solver_settings["abs_tol"].as<double>() : -1.0;
            qp_settings_.verbose = (solver_settings["verbose"]) && solver_settings["verbose"].as<bool>();
            qp_settings_.polish = (solver_settings["polish"]) && solver_settings["polish"].as<bool>();
            qp_settings_.rho = (solver_settings["rho"]) ? solver_settings["rho"].as<double>() : -1.0;
            qp_settings_.alpha = (solver_settings["alpha"]) ? solver_settings["alpha"].as<double>() : -1.0;
            qp_settings_.adaptive_rho = (solver_settings["adaptive_rho"]) && solver_settings["adaptive_rho"].as<bool>();
            qp_settings_.max_iter = (solver_settings["max_iter"]) ? solver_settings["max_iter"].as<int>() : -1;

            qp_solver_.UpdateSettings(qp_settings_);
        }

        // Assign to proper values

        // Assign settings for OSQP

    }

    void WholeBodyQPController::UpdateTargetState(const vectorx_t& target_state) {
        target_state_ = target_state;
    }

    vectorx_t WholeBodyQPController::ComputeControl(const vectorx_t& state, const models::Contact& contact) {

        // --------- Constraints --------- //
        // Dynamics constraints

        // Holonomic constraints

        // Torque constraints

        // Friction cone constraints

        // Positive GRF constraints

        // --------- Costs --------- //
        // Leg tracking costs

        // Torso tracking costs

        // Force tracking costs

        // --------- Update QP & Solve --------- //

        return vectorx_t::Zero(1);
    }

    // ---------------------------------------- //
    // --------- Constraint Functions --------- //
    // ---------------------------------------- //


}