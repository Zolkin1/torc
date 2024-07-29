//
// Created by zolkin on 7/28/24.
//
#include <iostream>

#include "yaml-cpp/yaml.h"

#include "full_order_mpc.h"
#include "eigen_utils.h"

// TODO: Figure out how I want to handle the geometry stuff

namespace torc::mpc {
    FullOrderMpc::FullOrderMpc(const fs::path& config_file, const fs::path& model_path)
        : config_file_(config_file), verbose_(true) {
        // Verify the robot file exists
        if (!fs::exists(model_path)) {
            throw std::runtime_error("Robot file does not exist!");
        }
        robot_model_ = std::make_unique<models::FullOrderRigidBody>("mpc_robot", model_path);

        // Verify the config file exists
        if (!fs::exists(config_file_)) {
            throw std::runtime_error("Configuration file does not exist!");
        }
        UpdateConfigurations();

        ws_ = std::make_unique<Workspace>();
    }

    void FullOrderMpc::UpdateConfigurations() {
        // Read in the yaml file.
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
                nodes_ = general_settings["nodes"].as<int>();
            } else {
                throw std::runtime_error("Number of nodes not specified!");
            }

            if (general_settings["verbose"]) {
                verbose_ = general_settings["verbose"].as<bool>();
            }
        }

        // ---------- Solver Settings ---------- //
        if (!config["solver_settings"]) {
            if (verbose_) {
                std::cout << "[MPC] No solver settings given. Using defaults." << std::endl;
            }
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

        // ---------- Constraint Settings ---------- //
        if (!config["constraints"]) {
            throw std::runtime_error("No constraint settings provided!");
        }
        YAML::Node constraint_settings = config["constraints"];
        friction_coef_ = constraint_settings["friction_coef"].as<double>();
        max_grf_ = constraint_settings["max_grf"].as<double>();

        // ---------- Cost Settings ---------- //
        if (!config["costs"]) {
            throw std::runtime_error("No cost settings provided!");
        }
        YAML::Node cost_settings = config["costs"];

        // ---------- Contact Settings ---------- //
        if (!config["contacts"]) {
            throw std::runtime_error("No contact settings provided!");
        }
        YAML::Node contact_settings = config["contacts"];
        num_contact_locations_ = contact_settings["num_contact_locations"].as<int>();
        contact_frames_ = contact_settings["contact_frames"].as<std::vector<std::string>>();

        if (verbose_) {
            using std::setw;
            using std::setfill;

            const int total_width = 50;

            solvers::OSQPInterfaceSettings qp_settings_ = qp_solver_.GetSettings();

            auto time_now = std::chrono::system_clock::now();
            std::time_t time1_now = std::chrono::system_clock::to_time_t(time_now);

            std::cout << setfill('=') << setw(total_width/2 - 7) << "" << " MPC Settings " << setw(total_width/2 - 7) << "" << std::endl;
            std::cout << "Current time: " << std::ctime(&time1_now);

            std::cout << "General settings: " << std::endl;
            std::cout << "\tVerbose: " << (verbose_ ? "True" : "False") << std::endl;
            std::cout << "\tNodes: " << nodes_ << std::endl;

            std::cout << "Solver settings: " << std::endl;
            std::cout << "\tRelative tolerance: " << qp_settings_.rel_tol << std::endl;
            std::cout << "\tAbsolute tolerance: " << qp_settings_.abs_tol << std::endl;
            std::cout << "\tVerbose: " << ((qp_settings_.verbose == 1) ? "True" : "False") << std::endl;
            std::cout << "\tPolish: " << ((qp_settings_.polish == 1) ? "True" : "False") << std::endl;
            std::cout << "\trho: " << qp_settings_.rho << std::endl;
            std::cout << "\talpha: " << qp_settings_.alpha << std::endl;
            std::cout << "\tAdaptive rho: " << qp_settings_.adaptive_rho << std::endl;
            std::cout << "\tMax iterations: " << qp_settings_.max_iter << std::endl;

            std::cout << "Constraints:" << std::endl;
            std::cout << "\tFriction coefficient: " << friction_coef_ << std::endl;
            std::cout << "\tMaximum ground reaction force: " << max_grf_ << std::endl;

            std::cout << "Costs:" << std::endl << std::endl;

            std::cout << "Contacts:" << std::endl;
            std::cout << "\tNumber of contact locations: " << num_contact_locations_ << std::endl;
            std::cout << "\tContact frames: [ ";
            for (const auto& frame : contact_frames_) {
                std::cout << frame << " ";
            }
            std::cout << "]" << std::endl;

            std::cout << "Size: " << std::endl;
            std::cout << "\tDecision variables: " << GetNumDecisionVars() << std::endl;
            std::cout << "\tConstraints: " << GetNumConstraints() << std::endl;

            std::cout << setfill('=') << setw(total_width) << "" << std::endl;

        }
    }

    void FullOrderMpc::ConfigureMpc() {
        // Create the constraint matrix
        constraints_.lb.resize(GetNumConstraints());
        constraints_.ub.resize(GetNumConstraints());
        constraints_.A.resize(GetNumConstraints(), GetNumDecisionVars());

        // Create A sparsity pattern to configure OSQP with
        CreateConstraintSparsityPattern();

        // Reset the triplet index, so we can re-use the triplet vector without re-allocating
        triplet_idx_ = 0;
    }

    // ------------------------------------------------- //
    // ----------- Sparsity Pattern Creation ----------- //
    // ------------------------------------------------- //
    void FullOrderMpc::CreateConstraintSparsityPattern() {
        // Fill out the triplets with dummy values -- this will allocate all the memory for the triplets
        for (int node = 0; node < nodes_ - 1; node++) {
            // Dynamics related constraints don't happen in the last node
            AddIntegrationPattern(node);
            AddIDPattern(node);
            AddFrictionConePattern(node);
            AddConfigurationBoxPattern();
            AddVelocityBoxPattern();
            AddTorqueBoxPattern();
            AddSwingHeightPattern();
            AddHolonomicPattern();
        }

        AddFrictionConePattern(nodes_ - 1);
        AddConfigurationBoxPattern();
        AddVelocityBoxPattern();
        AddTorqueBoxPattern();
        AddSwingHeightPattern();
        AddHolonomicPattern();

        // Make the matrix with the sparsity pattern
        constraints_.A.setFromTriplets(constraint_triplets_.begin(), constraint_triplets_.end());
    }

    void FullOrderMpc::AddIntegrationPattern(int node) {
        assert(node != nodes_ - 1);

        const int row_start = GetConstraintRow(node, Integrator);

        // q_k identity
        int col_start = GetDecisionIdx(node, Configuration);
        ws_->int_mat.setIdentity(robot_model_->GetVelDim(), robot_model_->GetVelDim());
        MatrixToNewTriplet(ws_->int_mat, row_start, col_start);

        // q_k+1 negative identity
        col_start = GetDecisionIdx(node + 1, Configuration);
        MatrixToNewTriplet(ws_->int_mat, row_start, col_start);

        // TODO: What is this sparsity pattern?
        // velocity mapping
        col_start = GetDecisionIdx(node, Velocity);
        ws_->int_mat.setConstant(1);
        MatrixToNewTriplet(ws_->int_mat, row_start, col_start);
    }

    void FullOrderMpc::AddIDPattern(int node) {
        assert(node != nodes_ - 1);

        const int row_start = GetConstraintRow(node, Integrator);

        // dtau_dq
        int col_start = GetDecisionIdx(node, Configuration);
        ws_->id_state_mat.setConstant(robot_model_->GetNumInputs() + FLOATING_VEL, robot_model_->GetVelDim(), 1);
        MatrixToNewTriplet(ws_->id_state_mat, row_start, col_start);

        // dtau_dv
        col_start = GetDecisionIdx(node, Velocity);
        MatrixToNewTriplet(ws_->id_state_mat, row_start, col_start);

        // dtau_dtau
        col_start = GetDecisionIdx(node, Torque);
        matrixx_t id;
        id.setIdentity(robot_model_->GetNumInputs(), robot_model_->GetNumInputs());
        MatrixToNewTriplet(id, row_start + FLOATING_VEL, col_start);

        // dtau_df
        col_start = GetDecisionIdx(node, GroundForce);
        MatrixToNewTriplet(ws_->id_state_mat, row_start, col_start);

        // dtau_dv2
        col_start = GetDecisionIdx(node + 1, Velocity);
        MatrixToNewTriplet(ws_->id_state_mat, row_start, col_start);
    }

    void FullOrderMpc::AddFrictionConePattern(int node) {
        const int row_start = GetConstraintRow(node, Integrator);

        // No GRF constraints when in swing
        int col_start = GetDecisionIdx(node, GroundForce);
        matrixx_t id;
        id.setIdentity(CONTACT_3DOF, CONTACT_3DOF);
        MatrixToNewTriplet(id, row_start, col_start);

        ws_->fric_cone_mat.setConstant(4, 3, 1);
        MatrixToNewTriplet(id, row_start + CONTACT_3DOF, col_start);
    }

    // ------------------------------------------------- //
    // ---------------- Helper Functions --------------- //
    // ------------------------------------------------- //
    int FullOrderMpc::GetNumConstraints() const {
        // Need to account for the fact that the last node is different
        // The last node does NOT have a ID or integrator constraint
        return GetConstraintsPerNode() * nodes_ - (NumIntegratorConstraintsNode() + NumIDConstraintsNode());
    }

    int FullOrderMpc::GetConstraintsPerNode() const {
        // Calc number of constraints per node then multiply by number of nodes
       return NumIntegratorConstraintsNode()
             + NumIDConstraintsNode()
             + NumFrictionConeConstraintsNode()
             + NumConfigBoxConstraintsNode()
             + NumVelocityBoxConstraintsNode()
             + NumTorqueBoxConstraintsNode()
             + NumSwingHeightConstraintsNode()
             + NumHolonomicConstraintsNode();
    }

    int FullOrderMpc::GetNumDecisionVars() const {
        return GetDecisionVarsPerNode() * nodes_;
    }

    int FullOrderMpc::GetDecisionVarsPerNode() const {
        return 2*robot_model_->GetVelDim() + robot_model_->GetNumInputs() + num_contact_locations_*CONTACT_3DOF;
    }

    int FullOrderMpc::GetDecisionIdx(int node, const DecisionType& var_type) const {
        int idx = GetDecisionVarsPerNode()*node;
        switch (var_type) {
            case GroundForce:
                idx += robot_model_->GetNumInputs();
            case Torque:
                idx += robot_model_->GetVelDim();
            case Velocity:
                idx += robot_model_->GetVelDim();
            case Configuration:
                break;
        }
        return idx;
    }

    int FullOrderMpc::GetConstraintRow(int node, const ConstraintType& constraint) const {
        int row = GetConstraintsPerNode()*node;
        switch (constraint) {
            case Holonomic:
                row += NumSwingHeightConstraintsNode();
            case SwingHeight:
                row += NumTorqueBoxConstraintsNode();
            case TorqueBox:
                row += NumVelocityBoxConstraintsNode();
            case VelBox:
                row += NumConfigBoxConstraintsNode();
            case ConfigBox:
                row += NumFrictionConeConstraintsNode();
            case FrictionCone:
                row += NumIDConstraintsNode();
            case ID:
                row += NumIntegratorConstraintsNode();
            case Integrator:
                break;
        }

        return row;
    }


    void FullOrderMpc::MatrixToNewTriplet(const torc::mpc::matrixx_t& mat, int row_start, int col_start) {
        for (int row = 0; row < mat.rows(); row++) {
            for (int col = 0; col < mat.cols(); col++) {
                // Only in this function do we want to filter out 0's because if they occur here then they are structural
                if (mat(row, col) != 0) {
                    constraint_triplets_.emplace_back(row_start + row, col_start + col, mat(row, col));
                }
            }
        }
    }

    void FullOrderMpc::MatrixToTriplet(const torc::mpc::matrixx_t& mat, int row_start, int col_start) {
        for (int row = 0; row < mat.rows(); row++) {
            for (int col = 0; col < mat.cols(); col++) {
                // Don't filter zero's here as they aren't structural
                constraint_triplets_[triplet_idx_] = Eigen::Triplet<double>(row_start + row, col_start + col, mat(row, col));
                triplet_idx_++;
            }
        }
    }

    void FullOrderMpc::DiagonalMatrixToTriplet(const torc::mpc::matrixx_t& mat, int row_start, int col_start) {
        for (int idx = 0; idx < mat.rows(); idx++) {
            // Don't filter zero's here as they aren't structural
            constraint_triplets_[triplet_idx_] = Eigen::Triplet<double>(row_start + idx, col_start + idx, mat(idx, idx));
            triplet_idx_++;
        }
    }

    // ------------------------------------------------- //
    // ----- Getters for Sizes of Individual nodes ----- //
    // ------------------------------------------------- //
    int FullOrderMpc::NumIntegratorConstraintsNode() const {
        // For now, all the configurations will be represented as deltas in the tangent space
        // TODO: Check this
        return robot_model_->GetVelDim();
    }

    int FullOrderMpc::NumIDConstraintsNode() const {
        // Floating base plus the number of inputs
        return FLOATING_VEL + robot_model_->GetNumInputs();
    }

    int FullOrderMpc::NumFrictionConeConstraintsNode() const {
        return FRICTION_CONE_SIZE;
    }

    int FullOrderMpc::NumConfigBoxConstraintsNode() const {
        // For now, all the configurations will be represented as deltas in the tangent space
        // TODO: Check this
        return robot_model_->GetVelDim();
    }

    int FullOrderMpc::NumVelocityBoxConstraintsNode() const {
        return robot_model_->GetVelDim();
    }

    int FullOrderMpc::NumTorqueBoxConstraintsNode() const {
        return robot_model_->GetNumInputs();
    }

    int FullOrderMpc::NumSwingHeightConstraintsNode() const {
        return num_contact_locations_;
    }

    int FullOrderMpc::NumHolonomicConstraintsNode() const {
        return num_contact_locations_*2;
    }

} // namespace torc::mpc