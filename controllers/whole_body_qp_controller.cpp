//
// Created by zolkin on 7/17/24.
//

#include "whole_body_qp_controller.h"

namespace torc::controllers {
    WholeBodyQPController::WholeBodyQPController() = default;

    WholeBodyQPController::WholeBodyQPController(const fs::path& config_file_path)
        : config_file_path_(config_file_path) {
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

        // Assign to proper values
    }

    void WholeBodyQPController::UpdateTargetState(const vectorx_t& target_state) {
        target_state_ = target_state;
    }

    vectorx_t WholeBodyQPController::ComputeControl(const vectorx_t& state, const models::Contact& contact) {
        return vectorx_t::Zero(1);
    }
}