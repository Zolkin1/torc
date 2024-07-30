//
// Created by zolkin on 7/28/24.
//
#include <iostream>

#include "yaml-cpp/yaml.h"

#include "full_order_mpc.h"
#include "eigen_utils.h"
#include "torc_timer.h"

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
        traj_.UpdateSizes(robot_model_->GetConfigDim(), robot_model_->GetVelDim(),
                          robot_model_->GetNumInputs(), contact_frames_, nodes_);
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

            if (general_settings["node_dt"]) {
                const auto dt = general_settings["node_dt"].as<double>();
                dt_.resize(nodes_ - 1);
                for (double & it : dt_) {
                    it = dt;
                }
            } else {
                throw std::runtime_error("Node dt not specified!");
            }
        }

        // ---------- Solver Settings ---------- //
        if (!config["solver_settings"]) {
            if (verbose_) {
                std::cout << "[MPC] No solver settings given. Using defaults." << std::endl;
            }
        } else {
            YAML::Node solver_settings = config["solver_settings"];
            osqp_settings_.eps_rel = (solver_settings["rel_tol"]) ? solver_settings["rel_tol"].as<double>() : -1.0;
            osqp_settings_.eps_abs = (solver_settings["abs_tol"]) ? solver_settings["abs_tol"].as<double>() : -1.0;
            osqp_settings_.verbose = (solver_settings["verbose"]) && solver_settings["verbose"].as<bool>();
            osqp_settings_.polish = (solver_settings["polish"]) && solver_settings["polish"].as<bool>();
            if (solver_settings["rho"]) {
                osqp_settings_.rho = solver_settings["rho"].as<double>();
            }
            if (solver_settings["alpha"]) {
                osqp_settings_.alpha = solver_settings["alpha"].as<double>();
            }
            osqp_settings_.adaptive_rho = (solver_settings["adaptive_rho"]) && solver_settings["adaptive_rho"].as<bool>();
            if (solver_settings["max_iter"]) {
                osqp_settings_.max_iter = solver_settings["max_iter"].as<int>();
            }
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
        contact_frames_ = contact_settings["contact_frames"].as<std::vector<std::string>>();
        num_contact_locations_ = contact_frames_.size();

        if (verbose_) {
            using std::setw;
            using std::setfill;

            const int total_width = 50;


            auto time_now = std::chrono::system_clock::now();
            std::time_t time1_now = std::chrono::system_clock::to_time_t(time_now);

            std::cout << setfill('=') << setw(total_width/2 - 7) << "" << " MPC Settings " << setw(total_width/2 - 7) << "" << std::endl;
            std::cout << "Current time: " << std::ctime(&time1_now);

            std::cout << "General settings: " << std::endl;
            std::cout << "\tVerbose: " << (verbose_ ? "True" : "False") << std::endl;
            std::cout << "\tNodes: " << nodes_ << std::endl;

            std::cout << "Solver settings: " << std::endl;
            std::cout << "\tRelative tolerance: " << osqp_settings_.eps_rel << std::endl;
            std::cout << "\tAbsolute tolerance: " << osqp_settings_.eps_abs << std::endl;
            std::cout << "\tVerbose: " << (osqp_settings_.verbose ? "True" : "False") << std::endl;
            std::cout << "\tPolish: " << (osqp_settings_.polish ? "True" : "False") << std::endl;
            std::cout << "\trho: " << osqp_settings_.rho << std::endl;
            std::cout << "\talpha: " << osqp_settings_.alpha << std::endl;
            std::cout << "\tAdaptive rho: " << osqp_settings_.adaptive_rho << std::endl;
            std::cout << "\tMax iterations: " << osqp_settings_.max_iter << std::endl;

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

    void FullOrderMpc::Configure() {
        utils::TORCTimer config_timer;
        config_timer.Tic();
        // Create the constraint matrix
        osqp_instance_.constraint_matrix.resize(GetNumConstraints(), GetNumDecisionVars());
        osqp_instance_.lower_bounds.resize(GetNumConstraints());
        osqp_instance_.upper_bounds.resize(GetNumConstraints());
        osqp_instance_.objective_vector.resize(GetNumDecisionVars());
        osqp_instance_.objective_matrix.resize(GetNumDecisionVars(), GetNumDecisionVars());

        osqp_instance_.objective_matrix.setIdentity();
        osqp_instance_.objective_vector.setConstant(2);
        osqp_instance_.lower_bounds.setConstant(-1);
        osqp_instance_.upper_bounds.setConstant(1);

        // Create A sparsity pattern to configure OSQP with
        CreateConstraintSparsityPattern();

        // Copy into constraints, which is what will be updated throughout the code
        // FromTriplets destroys the object, and osqp-cpp uses a reference to osqp_instance_, so we can't destroy it.
        // Thus, we will use constraints_ to hold the new constraints then copy the data vector in.
        // I believe we can update the bounds in place though.
        A = osqp_instance_.constraint_matrix;

        // Init OSQP
        auto status = osqp_solver_.Init(osqp_instance_, osqp_settings_);    // Takes about 5ms for 20 nodes

        // Reset the triplet index, so we can re-use the triplet vector without re-allocating
        triplet_idx_ = 0;

        // Setup trajectory
        traj_.SetNumNodes(nodes_);
        traj_.SetDefault(robot_model_->GetNeutralConfig());

        // Setup remaining workspace memory
        ws_->acc.resize(robot_model_->GetVelDim());
        for (const auto& frame : contact_frames_) {
            ws_->f_ext.emplace_back(frame, vector3_t::Zero());
        }

        config_timer.Toc();
        if (verbose_) {
            std::cout << "MPC configuration took " << config_timer.Duration<std::chrono::microseconds>().count()/1000.0 << "ms." << std::endl;
        }
    }

    Trajectory FullOrderMpc::Compute(const vectorx_t& state) {
        utils::TORCTimer compute_timer;
        compute_timer.Tic();

        vectorx_t q, v;
        robot_model_->ParseState(state, q, v);

        traj_.SetConfiguration(0, q);
        traj_.SetVelocity(0, v);

        CreateConstraints();
//        assert(triplet_idx_ == constraint_triplets_.size());

        CreateCost();

        // Update OSQP
        osqp_solver_.UpdateConstraintMatrix(osqp_instance_.constraint_matrix);
        auto status = osqp_solver_.Solve();

        compute_timer.Toc();
        if (verbose_) {
            std::cout << "MPC compute took " << compute_timer.Duration<std::chrono::microseconds>().count()/1000.0 << "ms." << std::endl;
        }

        return traj_;
    }

    // ------------------------------------------------- //
    // -------------- Constraint Creation -------------- //
    // ------------------------------------------------- //
    void FullOrderMpc::CreateConstraints() {
        triplet_idx_ = 0;
        AddICConstraint();
        for (int node = 0; node < nodes_ - 1; node++) {
            // Dynamics related constraints don't happen in the last node
            AddIntegrationConstraint(node);
            AddIDConstraint(node);
            AddFrictionConeConstraint(node);
            AddConfigurationBoxConstraint(node);
            AddVelocityBoxConstraint(node);
            AddTorqueBoxConstraint(node);
//            AddSwingHeightConstraint(node);
//            AddHolonomicConstraint(node);
        }

        AddFrictionConeConstraint(nodes_ - 1);
        AddConfigurationBoxConstraint(nodes_ - 1);
        AddVelocityBoxConstraint(nodes_ - 1);
        AddTorqueBoxConstraint(nodes_ - 1);
//        AddSwingHeightConstraint(nodes_ - 1);
//        AddHolonomicConstraint(nodes_ - 1);
    }

    void FullOrderMpc::AddICConstraint() {
        // Set constraint matrix
        int row_start = 0;
        int col_start = 0;
        DiagonalScalarMatrixToTriplet(1, row_start, col_start, robot_model_->GetVelDim());

        row_start += robot_model_->GetVelDim();
        col_start += robot_model_->GetVelDim();
        DiagonalScalarMatrixToTriplet(1, row_start, col_start, robot_model_->GetVelDim());

        // Set bounds
        // TODO: Need to convert these configurations into the proper tangent space!
//        osqp_instance_.lower_bounds.head(robot_model_->GetVelDim()) = traj_.GetConfiguration(0);
//        osqp_instance_.upper_bounds.head(robot_model_->GetVelDim()) = traj_.GetConfiguration(0);

        osqp_instance_.lower_bounds.segment(robot_model_->GetVelDim(), robot_model_->GetVelDim()) = traj_.GetVelocity(0);
        osqp_instance_.upper_bounds.segment(robot_model_->GetVelDim(), robot_model_->GetVelDim()) = traj_.GetVelocity(0);
    }

    void FullOrderMpc::AddIntegrationConstraint(int node) {
        assert(node != nodes_ - 1);

        const int row_start = GetConstraintRow(node, Integrator);

        // q_k identity
        int col_start = GetDecisionIdx(node, Configuration);
        DiagonalScalarMatrixToTriplet(1, row_start, col_start, robot_model_->GetVelDim());

        // q_k+1 negative identity
        col_start = GetDecisionIdx(node + 1, Configuration);
        DiagonalScalarMatrixToTriplet(-1, row_start, col_start, robot_model_->GetVelDim());

        // TODO: What is this sparsity pattern?
        // velocity mapping
        // TODO: SET!!
        col_start = GetDecisionIdx(node, Velocity);
        ws_->int_mat.setConstant(1);
        MatrixToTriplet(ws_->int_mat, row_start, col_start);
        // TODO: SET!!

        // TODO: Set the bounds
    }

    void FullOrderMpc::AddIDConstraint(int node) {
        assert(node != nodes_ - 1);

        ws_->acc = (traj_.GetVelocity(node + 1) - traj_.GetVelocity(node))/dt_[node];

        const int row_start = GetConstraintRow(node, ID);

        // dtau_dq
        int col_start = GetDecisionIdx(node, Configuration);
        ws_->id_state_mat.setConstant(robot_model_->GetNumInputs() + FLOATING_VEL, robot_model_->GetVelDim(), 1);
        MatrixToTriplet(ws_->id_state_mat, row_start, col_start);

        // dtau_dv
        col_start = GetDecisionIdx(node, Velocity);
        MatrixToTriplet(ws_->id_state_mat, row_start, col_start);

        // dtau_dtau
        col_start = GetDecisionIdx(node, Torque);
        DiagonalScalarMatrixToTriplet(-1, row_start + FLOATING_VEL, col_start, robot_model_->GetNumInputs());

        // dtau_df
        col_start = GetDecisionIdx(node, GroundForce);
        ws_->id_force_mat.setConstant(robot_model_->GetNumInputs() + FLOATING_VEL, num_contact_locations_*CONTACT_3DOF, 1);
        MatrixToTriplet(ws_->id_force_mat, row_start, col_start);

        // dtau_dv2
        col_start = GetDecisionIdx(node + 1, Velocity);
        MatrixToTriplet(ws_->id_state_mat, row_start, col_start);

        // Set the bounds
        for (auto& f : ws_->f_ext) {
            f.force_linear = traj_.GetForce(node, f.frame_name);
        }

        osqp_instance_.lower_bounds.segment(row_start, robot_model_->GetNumInputs() + FLOATING_VEL)
            = robot_model_->InverseDynamics(traj_.GetConfiguration(node), traj_.GetVelocity(node), ws_->acc, ws_->f_ext);
        osqp_instance_.lower_bounds.segment(row_start + FLOATING_VEL, robot_model_->GetNumInputs()) -= traj_.GetTau(node);

        osqp_instance_.upper_bounds.segment(row_start, robot_model_->GetNumInputs() + FLOATING_VEL) =
                osqp_instance_.lower_bounds.segment(row_start, robot_model_->GetNumInputs() + FLOATING_VEL);

    }

    void FullOrderMpc::AddFrictionConeConstraint(int node) {
        int row_start = GetConstraintRow(node, FrictionCone);
        int col_start = GetDecisionIdx(node, GroundForce);

        vector3_t h = {1, 0, 0};
        vector3_t l = {0, 1, 0};
        vector3_t n = {0, 0, 1};

        ws_->fric_cone_mat << (h - n*friction_coef_).transpose(),
                -(h + n*friction_coef_).transpose(),
                (l - n*friction_coef_).transpose(),
                -(l + n*friction_coef_).transpose();

        for (int contact = 0; contact < num_contact_locations_; contact++) {
            // Setting force to zero when in swing
            DiagonalScalarMatrixToTriplet(1, row_start, col_start, CONTACT_3DOF);
            osqp_instance_.lower_bounds.segment(row_start, CONTACT_3DOF).setZero();
            osqp_instance_.upper_bounds.segment(row_start, CONTACT_3DOF).setZero();

            row_start += CONTACT_3DOF;

            // Force in friction cone when in contact
            MatrixToTriplet(ws_->fric_cone_mat, row_start, col_start);
            osqp_instance_.lower_bounds.segment(row_start, FRICTION_CONE_SIZE).setConstant(-std::numeric_limits<double>::max());
            osqp_instance_.upper_bounds.segment(row_start, FRICTION_CONE_SIZE).setZero();

            row_start += FRICTION_CONE_SIZE;
            col_start += CONTACT_3DOF;
        }
    }

    void FullOrderMpc::AddConfigurationBoxConstraint(int node) {
        const int row_start = GetConstraintRow(node, ConfigBox);
        const int col_start = GetDecisionIdx(node, Configuration);

        DiagonalScalarMatrixToTriplet(1, row_start, col_start, robot_model_->GetVelDim());

        // TODO: Add bounds
        // Get these with a call to pinocchio's model's upperPositionLimit and lowerPositionLimit
//        osqp_instance_.lower_bounds.segment(row_start, robot_model_->GetVelDim()).setConstant();
//        osqp_instance_.upper_bounds.segment(row_start, robot_model_->GetVelDim()).setZero();

    }

    void FullOrderMpc::AddVelocityBoxConstraint(int node) {
        const int row_start = GetConstraintRow(node, VelBox);
        const int col_start = GetDecisionIdx(node, Velocity);

        DiagonalScalarMatrixToTriplet(1, row_start, col_start, robot_model_->GetVelDim());

        // TODO: Add bounds
        // Get these with a call to pinocchio's model's velocityLimit
//        osqp_instance_.lower_bounds.segment(row_start, robot_model_->GetVelDim()).setConstant();
//        osqp_instance_.upper_bounds.segment(row_start, robot_model_->GetVelDim()).setZero();
    }

    void FullOrderMpc::AddTorqueBoxConstraint(int node) {
        const int row_start = GetConstraintRow(node, TorqueBox);
        const int col_start = GetDecisionIdx(node, Torque);

        DiagonalScalarMatrixToTriplet(1, row_start, col_start, robot_model_->GetNumInputs());

        // TODO: Add bounds
        // Get these with a call to pinocchio's model's effort limit
//        osqp_instance_.lower_bounds.segment(row_start, robot_model_->GetVelDim()).setConstant();
//        osqp_instance_.upper_bounds.segment(row_start, robot_model_->GetVelDim()).setZero();
    }

    void FullOrderMpc::AddSwingHeightConstraint(int node) {

    }

    void FullOrderMpc::AddHolonomicConstraint(int node) {

    }

    // ------------------------------------------------- //
    // ----------------- Cost Creation ----------------- //
    // ------------------------------------------------- //
    void FullOrderMpc::CreateCost() {

    }

    // ------------------------------------------------- //
    // ----------- Sparsity Pattern Creation ----------- //
    // ------------------------------------------------- //
    void FullOrderMpc::CreateConstraintSparsityPattern() {
        // Fill out the triplets with dummy values -- this will allocate all the memory for the triplets
        AddICPattern();

        for (int node = 0; node < nodes_ - 1; node++) {
            // Dynamics related constraints don't happen in the last node
            AddIntegrationPattern(node);
            AddIDPattern(node);
            AddFrictionConePattern(node);
            AddConfigurationBoxPattern(node);
            AddVelocityBoxPattern(node);
            AddTorqueBoxPattern(node);
            AddSwingHeightPattern(node);
            AddHolonomicPattern(node);
        }

        AddFrictionConePattern(nodes_ - 1);
        AddConfigurationBoxPattern(nodes_ - 1);
        AddVelocityBoxPattern(nodes_ - 1);
        AddTorqueBoxPattern(nodes_ - 1);
        AddSwingHeightPattern(nodes_ - 1);
        AddHolonomicPattern(nodes_ - 1);

//        int max_row = -1;
//        int max_col = -1;
//        for (const auto& trip : constraint_triplets_) {
//            if (trip.col() > max_col) {
//                max_col = trip.col();
//            }
//            if (trip.row() > max_row) {
//                max_row = trip.row();
//            }
//        }
//
//        std::cout << "max row: " << max_row << std::endl;
//        std::cout << "max col: " << max_col << std::endl;

        // Make the matrix with the sparsity pattern
        osqp_instance_.constraint_matrix.setFromTriplets(constraint_triplets_.begin(), constraint_triplets_.end());
    }

    void FullOrderMpc::AddICPattern() {
        int row_start = 0;
        int col_start = 0;
        matrixx_t id;
        id.setIdentity(robot_model_->GetVelDim(), robot_model_->GetVelDim());
        MatrixToNewTriplet(id, row_start, col_start);

        row_start += robot_model_->GetVelDim();
        col_start += robot_model_->GetVelDim();

        MatrixToNewTriplet(id, row_start, col_start);
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

        const int row_start = GetConstraintRow(node, ID);

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
        ws_->id_force_mat.setConstant(robot_model_->GetNumInputs() + FLOATING_VEL, num_contact_locations_*CONTACT_3DOF, 1);
        MatrixToNewTriplet(ws_->id_force_mat, row_start, col_start);

        // dtau_dv2
        col_start = GetDecisionIdx(node + 1, Velocity);
        MatrixToNewTriplet(ws_->id_state_mat, row_start, col_start);
    }

    void FullOrderMpc::AddFrictionConePattern(int node) {
        int row_start = GetConstraintRow(node, FrictionCone);
        int col_start = GetDecisionIdx(node, GroundForce);

        matrixx_t id;
        id.setIdentity(CONTACT_3DOF, CONTACT_3DOF);

        vector3_t h = {1, 0, 0};
        vector3_t l = {0, 1, 0};
        vector3_t n = {0, 0, 1};

        ws_->fric_cone_mat.resize(FRICTION_CONE_SIZE, CONTACT_3DOF);
        ws_->fric_cone_mat << (h - n*friction_coef_).transpose(),
                -(h + n*friction_coef_).transpose(),
                (l - n*friction_coef_).transpose(),
                -(l + n*friction_coef_).transpose();

        for (int contact = 0; contact < num_contact_locations_; contact++) {
            // Setting force to zero when in swing
            MatrixToNewTriplet(id, row_start, col_start);

            row_start += CONTACT_3DOF;

            // Force in friction cone when in contact
            MatrixToNewTriplet(ws_->fric_cone_mat, row_start, col_start);

            row_start += FRICTION_CONE_SIZE;
            col_start += CONTACT_3DOF;
        }
    }

    void FullOrderMpc::AddConfigurationBoxPattern(int node) {
        const int row_start = GetConstraintRow(node, ConfigBox);
        const int col_start = GetDecisionIdx(node, Configuration);

        matrixx_t id;
        id.setIdentity(robot_model_->GetVelDim(), robot_model_->GetVelDim());
        MatrixToNewTriplet(id, row_start, col_start);
    }

    void FullOrderMpc::AddVelocityBoxPattern(int node) {
        const int row_start = GetConstraintRow(node, VelBox);
        const int col_start = GetDecisionIdx(node, Velocity);

        matrixx_t id;
        id.setIdentity(robot_model_->GetVelDim(), robot_model_->GetVelDim());
        MatrixToNewTriplet(id, row_start, col_start);
    }

    void FullOrderMpc::AddTorqueBoxPattern(int node) {
        const int row_start = GetConstraintRow(node, TorqueBox);
        const int col_start = GetDecisionIdx(node, Torque);

        matrixx_t id;
        id.setIdentity(robot_model_->GetNumInputs(), robot_model_->GetNumInputs());
        MatrixToNewTriplet(id, row_start, col_start);
    }

    void FullOrderMpc::AddSwingHeightPattern(int node) {
        int row_start = GetConstraintRow(node, SwingHeight);
        const int col_start = GetDecisionIdx(node, Configuration);

        ws_->swing_vec.setConstant(robot_model_->GetVelDim(), 1);

        for (int contact = 0; contact < num_contact_locations_; contact++) {
            VectorToNewTriplet(ws_->swing_vec, row_start, col_start);
            row_start++;
        }
    }

    void FullOrderMpc::AddHolonomicPattern(int node) {
        int row_start = GetConstraintRow(node, Holonomic);
        const int col_start = GetDecisionIdx(node, Velocity);

        ws_->holo_mat.setConstant(2, robot_model_->GetVelDim(), 1);

        for (int contact = 0; contact < num_contact_locations_; contact++) {
            MatrixToNewTriplet(ws_->holo_mat, row_start, col_start);
            row_start += 2;
        }
    }

    // ------------------------------------------------- //
    // ---------------- Helper Functions --------------- //
    // ------------------------------------------------- //
    int FullOrderMpc::GetNumConstraints() const {
        // Need to account for the fact that the last node is different
        // The last node does NOT have a ID or integrator constraint
        // The start also has an initial condition constraint
        return GetConstraintsPerNode() * nodes_ - (NumIntegratorConstraintsNode() + NumIDConstraintsNode())
            + robot_model_->GetVelDim()*2;
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
        row += 2*robot_model_->GetVelDim(); // Account for initial condition constraints
        if (node == nodes_ - 1) {
            row -= NumIntegratorConstraintsNode() + NumIDConstraintsNode();
        }
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

    void FullOrderMpc::VectorToNewTriplet(const vectorx_t& vec, int row_start, int col_start) {
        for (int col = 0; col < vec.size(); col++) {
            // Only in this function do we want to filter out 0's because if they occur here then they are structural
            if (vec(col) != 0) {
                constraint_triplets_.emplace_back(row_start, col_start + col, vec(col));
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

    void FullOrderMpc::DiagonalScalarMatrixToTriplet(double val, int row_start, int col_start, int size) {
        for (int idx = 0; idx < size; idx++) {
            constraint_triplets_[triplet_idx_] = Eigen::Triplet<double>(row_start + idx, col_start + idx, val);
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
        return num_contact_locations_ * (FRICTION_CONE_SIZE + CONTACT_3DOF);
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

    // Other functions
    void FullOrderMpc::SetVerbosity(bool verbose) {
        verbose_ = verbose;
    }

} // namespace torc::mpc