//
// Created by zolkin on 7/28/24.
//
#include <iostream>

#include "yaml-cpp/yaml.h"

#include "full_order_mpc.h"
#include "autodiff_fn.h"

#include <pinocchio/algorithm/kinematics-derivatives.hxx>
#include <Eigen/Dense>

#include "eigen_utils.h"
#include "torc_timer.h"

#define NAN_CHECKS 0

namespace torc::mpc {
    FullOrderMpc::FullOrderMpc(const fs::path& config_file, const fs::path& model_path)
        : config_file_(config_file), verbose_(true), cost_("full_order_mpc_cost"), compile_derivatves_(true) {
        // Verify the robot file exists
        if (!fs::exists(model_path)) {
            throw std::runtime_error("Robot file does not exist!");
        }
        robot_model_ = std::make_unique<models::FullOrderRigidBody>("mpc_robot", model_path);

        // Verify the config file exists
        if (!fs::exists(config_file_)) {
            throw std::runtime_error("Configuration file does not exist!");
        }
        UpdateSettings();

        ws_ = std::make_unique<Workspace>();
        traj_.UpdateSizes(robot_model_->GetConfigDim(), robot_model_->GetVelDim(),
                          robot_model_->GetNumInputs(), contact_frames_, nodes_);

        // CreateDefaultCost();
    }

    void FullOrderMpc::UpdateSettings() {
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

            if (general_settings["compile_derivatives"]) {
                compile_derivatves_ = general_settings["compile_derivatives"].as<bool>();
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
            if (solver_settings["adaptive_rho"]) {
                osqp_settings_.adaptive_rho = solver_settings["adaptive_rho"].as<bool>();
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
        if (cost_settings["configuration_tracking_weights"]) {
            auto config_tracking_weights = cost_settings["configuration_tracking_weights"].as<std::vector<double>>();
            config_tracking_weight_ = utils::StdToEigenVector(config_tracking_weights);
        }

        if (config_tracking_weight_.size() < robot_model_->GetVelDim()) {
            std::cerr << "Configuration tracking weight size is too small, adding zeros." <<
               "Expected size " << robot_model_->GetVelDim() << ", but got size " << config_tracking_weight_.size() << std::endl;
            int starting_size = config_tracking_weight_.size();
            config_tracking_weight_.conservativeResize(robot_model_->GetVelDim());
            for (int i = starting_size; i < robot_model_->GetVelDim(); i++) {
                config_tracking_weight_(i) = 0;
            }
        } else if (config_tracking_weight_.size() > robot_model_->GetVelDim()) {
            std::cerr << "Configuration tracking weight is too large. Ignoring end values." <<
               "Expected size " << robot_model_->GetVelDim() << ", but got size " << config_tracking_weight_.size() << std::endl;
        }

        if (cost_settings["velocity_tracking_weights"]) {
            auto vel_tracking_weights = cost_settings["velocity_tracking_weights"].as<std::vector<double>>();
            vel_tracking_weight_ = utils::StdToEigenVector(vel_tracking_weights);
        }

        if (vel_tracking_weight_.size() < robot_model_->GetVelDim()) {
            std::cerr << "Velocity tracking weight size is too small, adding zeros." <<
               "Expected size " << robot_model_->GetVelDim() << ", but got size " << vel_tracking_weight_.size() << std::endl;
            int starting_size = vel_tracking_weight_.size();
            vel_tracking_weight_.conservativeResize(robot_model_->GetVelDim());
            for (int i = starting_size; i < robot_model_->GetVelDim(); i++) {
                vel_tracking_weight_(i) = 0;
            }
        } else if (vel_tracking_weight_.size() > robot_model_->GetVelDim()) {
            std::cerr << "Velocity tracking weight is too large. Ignoring end values." <<
               "Expected size " << robot_model_->GetVelDim() << ", but got size " << vel_tracking_weight_.size() << std::endl;
        }

        std::cout << "vel tracking: " << vel_tracking_weight_.transpose() << std::endl;
        std::cout << "config tracking: " << config_tracking_weight_.transpose() << std::endl;
        // ---------- Contact Settings ---------- //
        if (!config["contacts"]) {
            throw std::runtime_error("No contact settings provided!");
        }
        YAML::Node contact_settings = config["contacts"];
        contact_frames_ = contact_settings["contact_frames"].as<std::vector<std::string>>();
        num_contact_locations_ = contact_frames_.size();
        for (const auto& frame : contact_frames_) {
            std::vector<double> swing_traj;
            swing_traj.resize(nodes_);
            swing_traj_.insert(std::pair<std::string, std::vector<double>>(frame, swing_traj));
        }

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
            std::cout << "\tCompile derivatives: " << (compile_derivatves_ ? "True" : "False") << std::endl;

            std::cout << "Solver settings: " << std::endl;
            std::cout << "\tRelative tolerance: " << osqp_settings_.eps_rel << std::endl;
            std::cout << "\tAbsolute tolerance: " << osqp_settings_.eps_abs << std::endl;
            std::cout << "\tVerbose: " << (osqp_settings_.verbose ? "True" : "False") << std::endl;
            std::cout << "\tPolish: " << (osqp_settings_.polish ? "True" : "False") << std::endl;
            std::cout << "\trho: " << osqp_settings_.rho << std::endl;
            std::cout << "\talpha: " << osqp_settings_.alpha << std::endl;
            std::cout << "\tAdaptive rho: " << (osqp_settings_.adaptive_rho ? "True" : "False") << std::endl;
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

            std::cout << "Robot: " << std::endl;
            std::cout << "\tName: " << robot_model_->GetUrdfRobotName() << std::endl;
            std::cout << "\tUpper Configuration Bounds: " << robot_model_->GetUpperConfigLimits().transpose() << std::endl;
            std::cout << "\tLower Configuration Bounds: " << robot_model_->GetLowerConfigLimits().transpose() << std::endl;
            std::cout << "\tVelocity Bounds: " << robot_model_->GetVelocityJointLimits().transpose() << std::endl;
            std::cout << "\tTorque Bounds: " << robot_model_->GetTorqueJointLimits().transpose() << std::endl;

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

        osqp_instance_.objective_matrix.setZero();
        osqp_instance_.objective_vector.setZero();
        osqp_instance_.lower_bounds.setConstant(-1);
        osqp_instance_.upper_bounds.setConstant(1);

        // Create A sparsity pattern to configure OSQP with
        CreateConstraintSparsityPattern();

        // Copy into constraints, which is what will be updated throughout the code
        // FromTriplets destroys the object, and osqp-cpp uses a reference to osqp_instance_, so we can't destroy it.
        // Thus, we will use constraints_ to hold the new constraints then copy the data vector in.
        // I believe we can update the bounds in place though.
        A_ = osqp_instance_.constraint_matrix;

        // Reset the triplet index, so we can re-use the triplet vector without re-allocating
        constraint_triplet_idx_ = 0;

        // Setup trajectory
        traj_.SetNumNodes(nodes_);
        traj_.SetDefault(robot_model_->GetNeutralConfig());

        // Setup remaining workspace memory
        ws_->acc.resize(robot_model_->GetVelDim());
        for (const auto& frame : contact_frames_) {
            ws_->f_ext.emplace_back(frame, vector3_t::Zero());
        }
        ws_->frame_jacobian.resize(6, robot_model_->GetVelDim());

        // Setup cost function
        // TODO: Move this
        std::vector<vectorx_t> weights;
        weights.emplace_back(config_tracking_weight_);
        weights.emplace_back(vel_tracking_weight_);

        std::vector<CostTypes> costs;
        costs.emplace_back(CostTypes::Configuration);
        costs.emplace_back(CostTypes::Velocity);
        cost_.Configure(robot_model_->GetConfigDim(), robot_model_->GetVelDim(), robot_model_->GetNumInputs(), compile_derivatves_,
        costs, weights);

        objective_mat_.resize(GetNumDecisionVars(), GetNumDecisionVars());
        CreateCostPattern();
        objective_mat_ = osqp_instance_.objective_matrix;

        // Init OSQP
        auto status = osqp_solver_.Init(osqp_instance_, osqp_settings_);    // Takes about 5ms for 20 nodes

        config_timer.Toc();
        if (verbose_) {
            std::cout << "MPC configuration took " << config_timer.Duration<std::chrono::microseconds>().count()/1000.0 << "ms." << std::endl;
        }
    }

    void FullOrderMpc::Compute(const vectorx_t& state, Trajectory& traj_out) {
        utils::TORCTimer compute_timer;
        compute_timer.Tic();

        vectorx_t q, v;
        robot_model_->ParseState(state, q, v);

        traj_.SetConfiguration(0, q);
        traj_.SetVelocity(0, v);

        CreateConstraints();
        for (auto& constraint_triplet : constraint_triplets_) {
            if(std::isnan(constraint_triplet.value())) {
                throw std::runtime_error("nan in constraint mat");
            }
        }

        for (int i = 0; i < osqp_instance_.lower_bounds.size(); i++) {
            if (std::isnan(osqp_instance_.lower_bounds(i))) {
                throw std::runtime_error("lower bound has nan at" + i);
            }

            if (std::isnan(osqp_instance_.upper_bounds(i))) {
                throw std::runtime_error("upper bound has nan at " + i);
            }
        }

        // TODO: Create q_target, v_target
        UpdateCost();
#if NAN_CHECKS
        for (const auto& objective_triplet : objective_triplets_) {
            if(std::isnan(objective_triplet.value())) {
                throw std::runtime_error("nan in constraint mat");
            }

            if (objective_triplet.value() == 0) {
                throw std::runtime_error("objective 0 in triplet!");
            }
        }
#endif

        // std::cout << "objective mat: \n" << objective_mat_ << std::endl;
        // std::cout << "objective vec: " << osqp_instance_.objective_vector.transpose() << std::endl;
        // std::cout << "A: \n" << A_ << std::endl;

        // Set matrices
        auto status = osqp_solver_.UpdateObjectiveAndConstraintMatrices(objective_mat_, A_);
        if (!status.ok()) {
            std::cerr << "status: " << status << std::endl;
            throw std::runtime_error("Could not update the objective and constraint matrix.");
        }

        // Set objective vector
        status = osqp_solver_.SetObjectiveVector(osqp_instance_.objective_vector);
        if (!status.ok()) {
            std::cerr << "status: " << status << std::endl;
            throw std::runtime_error("Could not update the objective vector.");
        }

        // Set upper and lower bounds
        status = osqp_solver_.SetBounds(osqp_instance_.lower_bounds, osqp_instance_.upper_bounds);
        if (!status.ok()) {
            std::cerr << "status: " << status << std::endl;
            throw std::runtime_error("Could not update the constraint bounds.");
        }

        // Solve
        auto solve_status = osqp_solver_.Solve();
        if (!status.ok()) {
            std::cerr << "status: " << status << std::endl;
            throw std::runtime_error("Could not solve the QP.");
        }

#if NAN_CHECKS
        for (const auto& res : osqp_solver_.primal_solution()) {
            if (std::isnan(res)) {
                throw std::runtime_error("nan in primal solution");
            }
        }

        for (const auto& res : osqp_solver_.dual_solution()) {
            if (std::isnan(res)) {
                throw std::runtime_error("nan in dual solution");
            }
        }
#endif

        ConvertSolutionToTraj(osqp_solver_.primal_solution(), traj_out);
        traj_ = traj_out;
        // std::cout << "solve result: \n" << osqp_solver_.primal_solution() << std::endl;

        double alpha = 0;

        compute_timer.Toc();
        stats_.emplace_back(solve_status, osqp_solver_.objective_value(),
            cost_.GetCost(osqp_solver_.primal_solution()), alpha, osqp_solver_.primal_solution().norm(),
            compute_timer.Duration<std::chrono::microseconds>().count()/1000.0);

        if (verbose_) {
            std::cout << "MPC compute took " << compute_timer.Duration<std::chrono::microseconds>().count()/1000.0 << "ms." << std::endl;
        }
    }

    // ------------------------------------------------- //
    // -------------- Constraint Creation -------------- //
    // ------------------------------------------------- //
    void FullOrderMpc::CreateConstraints() {
        constraint_triplet_idx_ = 0;
        AddICConstraint();
        for (int node = 0; node < nodes_ - 1; node++) {
            // std::cout << "node: " << node << std::endl;
            // std::cout << "config: " << traj_.GetConfiguration(node).transpose() << std::endl;
            // std::cout << "vel: " << traj_.GetVelocity(node).transpose() << std::endl;
            // std::cout << "torque: " << traj_.GetTau(node).transpose() << std::endl;

            // Dynamics related constraints don't happen in the last node
            AddIntegrationConstraint(node);
            AddIDConstraint(node);
            AddFrictionConeConstraint(node);
            AddTorqueBoxConstraint(node);

            // These could conflict with the initial condition constraints
            if (node > 1) {
                // Configuration is set for the initial condition and the next node, do not constrain it
                AddConfigurationBoxConstraint(node);
                AddSwingHeightConstraint(node);
            }
            if (node > 0) {
                // Velocity is fixed for the initial condition, do not constrain it
                AddHolonomicConstraint(node);
                AddVelocityBoxConstraint(node);
            }
        }

        AddFrictionConeConstraint(nodes_ - 1);
        AddConfigurationBoxConstraint(nodes_ - 1);
        AddVelocityBoxConstraint(nodes_ - 1);
        AddTorqueBoxConstraint(nodes_ - 1);
        AddSwingHeightConstraint(nodes_ - 1);
        AddHolonomicConstraint(nodes_ - 1);

        if (constraint_triplet_idx_ != constraint_triplets_.size()) {
            std::cerr << "triplet_idx: " << constraint_triplet_idx_ << "\nconstraint triplet size: " << constraint_triplets_.size() << std::endl;
            throw std::runtime_error("Constraints did not populate the full matrix!");
        }

        A_.setFromTriplets(constraint_triplets_.begin(), constraint_triplets_.end());
    }

    void FullOrderMpc::AddICConstraint() {
        // Set constraint matrix
        int row_start = 0;
        int col_start = 0;
        DiagonalScalarMatrixToTriplet(1, row_start, col_start, 2*robot_model_->GetVelDim(), constraint_triplets_, constraint_triplet_idx_);

        // Set bounds
        // Configuration differences are all set to 0
        osqp_instance_.lower_bounds.head(robot_model_->GetVelDim()).setZero();
        osqp_instance_.upper_bounds.head(robot_model_->GetVelDim()).setZero();

        // Velocity differences are all set to 0
        osqp_instance_.lower_bounds.segment(robot_model_->GetVelDim(), robot_model_->GetVelDim()).setZero();
        osqp_instance_.upper_bounds.segment(robot_model_->GetVelDim(), robot_model_->GetVelDim()).setZero();
    }

    void FullOrderMpc::AddIntegrationConstraint(int node) {
        assert(node != nodes_ - 1);

        int row_start = GetConstraintRow(node, Integrator);

        // q^b_k identity
        int col_start = GetDecisionIdx(node, Configuration);
        DiagonalScalarMatrixToTriplet(1, row_start, col_start, POS_VARS, constraint_triplets_, constraint_triplet_idx_);

        // q^q_k linearization
        row_start += POS_VARS;
        col_start += POS_VARS;
        matrix3_t dxi = QuatIntegrationLinearizationXi(node);
        MatrixToTriplet(dxi, row_start, col_start, constraint_triplets_, constraint_triplet_idx_);

        // q^j_k identity
        row_start += 3;
        col_start += 3;
        DiagonalScalarMatrixToTriplet(1, row_start, col_start, robot_model_->GetNumInputs(), constraint_triplets_, constraint_triplet_idx_);   // TODO Should probably be num joints not inputs

        // q_k+1 negative identity
        row_start = GetConstraintRow(node, Integrator);
        col_start = GetDecisionIdx(node + 1, Configuration);
        DiagonalScalarMatrixToTriplet(-1, row_start, col_start, robot_model_->GetVelDim(), constraint_triplets_, constraint_triplet_idx_);

        // v^b_k dt*identity
        row_start = GetConstraintRow(node, Integrator);
        col_start = GetDecisionIdx(node, Velocity);
        DiagonalScalarMatrixToTriplet(dt_[node], row_start, col_start, POS_VARS, constraint_triplets_, constraint_triplet_idx_);

        // v^q_k linearization
        row_start += POS_VARS;
        col_start += POS_VARS;
        matrix3_t dv = QuatIntegrationLinearizationW(node);
        MatrixToTriplet(dv, row_start, col_start, constraint_triplets_, constraint_triplet_idx_);

        // v^j_k dt*identity
        row_start += 3;
        col_start += 3;
        DiagonalScalarMatrixToTriplet(dt_[node], row_start, col_start, robot_model_->GetNumInputs(), constraint_triplets_, constraint_triplet_idx_);   // TODO Should probably be num joints not inputs

        // Base position bounds
        row_start = GetConstraintRow(node, Integrator);
        const vector3_t pos_constant = -traj_.GetConfiguration(node + 1).head<POS_VARS>() + traj_.GetConfiguration(node).head<POS_VARS>() + dt_[node]*traj_.GetVelocity(node).head<POS_VARS>();
        osqp_instance_.lower_bounds.segment<POS_VARS>(row_start) = pos_constant;
        osqp_instance_.upper_bounds.segment<POS_VARS>(row_start) = pos_constant;
        row_start += POS_VARS;

        // Base orientation bounds
        osqp_instance_.lower_bounds.segment<3>(row_start).setZero();
        osqp_instance_.upper_bounds.segment<3>(row_start).setZero();
        row_start += 3;

        // Joint bounds
        osqp_instance_.lower_bounds.segment(row_start, robot_model_->GetNumInputs()) = // TODO Should probably be num joints not inputs
             traj_.GetConfiguration(node+1).tail(robot_model_->GetNumInputs())
            - traj_.GetConfiguration(node).tail(robot_model_->GetNumInputs())
            - dt_[node]*traj_.GetVelocity(node).tail(robot_model_->GetNumInputs());
        osqp_instance_.upper_bounds.segment(row_start, robot_model_->GetNumInputs()) =
            osqp_instance_.lower_bounds.segment(row_start, robot_model_->GetNumInputs());

        row_start = GetConstraintRow(node, Integrator);
        // std::cout << "velocity traj tail: " << traj_.GetVelocity(node).tail(robot_model_->GetNumInputs()).transpose() << std::endl;
        // std::cout << "next node traj tail: " << traj_.GetConfiguration(node+1).tail(robot_model_->GetNumInputs()).transpose() << std::endl;
        // std::cout << "lb integration: " << osqp_instance_.lower_bounds.segment(row_start, robot_model_->GetVelDim()).transpose() << std::endl;
        // std::cout << "ub integration: " << osqp_instance_.upper_bounds.segment(row_start, robot_model_->GetVelDim()).transpose() << std::endl;
    }

    void FullOrderMpc::AddIDConstraint(int node) {
        assert(node != nodes_ - 1);

        const int row_start = GetConstraintRow(node, ID);

        ws_->id_config_mat.setZero();
        ws_->id_vel1_mat.setZero();
        ws_->id_vel2_mat.setZero();
        ws_->id_force_mat.setZero();

        // compute all derivative terms
        InverseDynamicsLinearization(node, ws_->id_config_mat,
            ws_->id_vel1_mat, ws_->id_vel2_mat, ws_->id_force_mat);

        // std::cout << "config mat: \n" << ws_->id_config_mat << std::endl;
        // std::cout << "vel1 mat: \n" << ws_->id_vel1_mat << std::endl;
        // std::cout << "vel2 mat: \n" << ws_->id_vel2_mat << std::endl;
        // std::cout << "force mat: \n" << ws_->id_force_mat << std::endl;

        // dtau_dq
        int col_start = GetDecisionIdx(node, Configuration);
        MatrixToTriplet(ws_->id_config_mat, row_start, col_start, constraint_triplets_, constraint_triplet_idx_);

        // dtau_dv
        col_start = GetDecisionIdx(node, Velocity);
        MatrixToTriplet(ws_->id_vel1_mat, row_start, col_start, constraint_triplets_, constraint_triplet_idx_);

        // dtau_dtau
        col_start = GetDecisionIdx(node, Torque);
        DiagonalScalarMatrixToTriplet(-1, row_start + FLOATING_VEL, col_start, robot_model_->GetNumInputs(),
            constraint_triplets_, constraint_triplet_idx_);

        // dtau_df
        col_start = GetDecisionIdx(node, GroundForce);
        MatrixToTriplet(ws_->id_force_mat, row_start, col_start, constraint_triplets_, constraint_triplet_idx_);

        // dtau_dv2
        col_start = GetDecisionIdx(node + 1, Velocity);
        MatrixToTriplet(ws_->id_vel2_mat, row_start, col_start, constraint_triplets_, constraint_triplet_idx_);

        // Set the bounds
        for (auto& f : ws_->f_ext) {
            f.force_linear = traj_.GetForce(node, f.frame_name);
        }

        osqp_instance_.lower_bounds.segment(row_start, robot_model_->GetNumInputs() + FLOATING_VEL)
            = -robot_model_->InverseDynamics(traj_.GetConfiguration(node), traj_.GetVelocity(node), ws_->acc, ws_->f_ext);
        osqp_instance_.lower_bounds.segment(row_start + FLOATING_VEL, robot_model_->GetNumInputs()) += traj_.GetTau(node);

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
            DiagonalScalarMatrixToTriplet(1, row_start, col_start, CONTACT_3DOF, constraint_triplets_, constraint_triplet_idx_);
            osqp_instance_.lower_bounds.segment(row_start, CONTACT_3DOF).setZero();
            osqp_instance_.upper_bounds.segment(row_start, CONTACT_3DOF).setZero();

            row_start += CONTACT_3DOF;

            // Force in friction cone when in contact
            MatrixToTriplet(ws_->fric_cone_mat, row_start, col_start, constraint_triplets_, constraint_triplet_idx_, true);
            osqp_instance_.lower_bounds.segment(row_start, FRICTION_CONE_SIZE).setConstant(-std::numeric_limits<double>::max());
            osqp_instance_.upper_bounds.segment(row_start, FRICTION_CONE_SIZE).setZero();

            row_start += FRICTION_CONE_SIZE;
            col_start += CONTACT_3DOF;
        }
    }

    void FullOrderMpc::AddConfigurationBoxConstraint(int node) {
        int row_start = GetConstraintRow(node, ConfigBox);
        int col_start = GetDecisionIdx(node, Configuration);

        // pos identity
        DiagonalScalarMatrixToTriplet(1, row_start, col_start, POS_VARS, constraint_triplets_, constraint_triplet_idx_);

        // Configuration linearization
        row_start += POS_VARS;
        col_start += POS_VARS;
        matrix43_t q_lin = QuatLinearization(node);
        MatrixToTriplet(q_lin, row_start, col_start, constraint_triplets_, constraint_triplet_idx_);

        // joint identity
        row_start += QUAT_VARS;
        col_start += 3;
        DiagonalScalarMatrixToTriplet(1, row_start, col_start, robot_model_->GetNumInputs(), constraint_triplets_, constraint_triplet_idx_); // TODO Should probably be num joints not inputs

        // Set configuration bounds
        row_start = GetConstraintRow(node, ConfigBox);
        osqp_instance_.lower_bounds.segment(row_start, robot_model_->GetConfigDim())
            = robot_model_->GetLowerConfigLimits() - traj_.GetConfiguration(node);

        osqp_instance_.upper_bounds.segment(row_start, robot_model_->GetConfigDim())
            = robot_model_->GetUpperConfigLimits() - traj_.GetConfiguration(node);

        // std::cout << "lb: " << osqp_instance_.lower_bounds.segment(row_start, robot_model_->GetConfigDim()).transpose() << std::endl;
        // std::cout << "ub: " << osqp_instance_.upper_bounds.segment(row_start, robot_model_->GetConfigDim()).transpose() << std::endl;
    }

    void FullOrderMpc::AddVelocityBoxConstraint(int node) {
        const int row_start = GetConstraintRow(node, VelBox);
        const int col_start = GetDecisionIdx(node, Velocity);

        DiagonalScalarMatrixToTriplet(1, row_start, col_start, robot_model_->GetVelDim(), constraint_triplets_, constraint_triplet_idx_);

        // Set velocity bounds
        osqp_instance_.lower_bounds.segment(row_start, robot_model_->GetVelDim())
            = -robot_model_->GetVelocityJointLimits() - traj_.GetVelocity(node);
        osqp_instance_.upper_bounds.segment(row_start, robot_model_->GetVelDim())
            = robot_model_->GetVelocityJointLimits() - traj_.GetVelocity(node);
    }

    void FullOrderMpc::AddTorqueBoxConstraint(int node) {
        const int row_start = GetConstraintRow(node, TorqueBox);
        const int col_start = GetDecisionIdx(node, Torque);

        DiagonalScalarMatrixToTriplet(1, row_start, col_start, robot_model_->GetNumInputs(), constraint_triplets_, constraint_triplet_idx_);

        // Set torque bounds
        osqp_instance_.lower_bounds.segment(row_start, robot_model_->GetNumInputs()) = robot_model_->GetTorqueJointLimits() - traj_.GetTau(node);
        osqp_instance_.upper_bounds.segment(row_start, robot_model_->GetNumInputs()) = robot_model_->GetTorqueJointLimits() - traj_.GetTau(node);
    }

    void FullOrderMpc::AddSwingHeightConstraint(int node) {
        int row_start = GetConstraintRow(node, SwingHeight);
        const int col_start = GetDecisionIdx(node, Configuration);

        for (const auto& frame : contact_frames_) {
            SwingHeightLinearization(node, frame, ws_->frame_jacobian);

            // Grab just the z-height element
            ws_->swing_vec = ws_->frame_jacobian.row(2);
            VectorToTriplet(ws_->swing_vec, row_start, col_start, constraint_triplets_, constraint_triplet_idx_);

            // Get the frame position on the warm start trajectory
            robot_model_->FirstOrderFK(traj_.GetConfiguration(node));
            vector3_t frame_pos = robot_model_->GetFrameState(frame).placement.translation();

            // Set bounds
            osqp_instance_.lower_bounds(row_start)
                = -frame_pos(2) - swing_traj_[frame][node];
            osqp_instance_.upper_bounds(row_start)
                = -frame_pos(2) + swing_traj_[frame][node];

            // std::cout << "frame: " << frame << std::endl;
            // std::cout << "frame pos: " << frame_pos.transpose() << std::endl;
            // std::cout << "linearization: " << ws_->swing_vec.transpose() << std::endl;
            // std::cout << "lb: " << osqp_instance_.lower_bounds(row_start) << std::endl;
            // std::cout << "ub: " << osqp_instance_.upper_bounds(row_start) << std::endl;

            row_start++;
        }
    }

    void FullOrderMpc::AddHolonomicConstraint(int node) {
        int row_start = GetConstraintRow(node, Holonomic);

        for (const auto& frame : contact_frames_) {
            // Get velocity linearization
            HolonomicLinearizationv(node, frame, ws_->frame_jacobian);
            int col_start = GetDecisionIdx(node, Velocity);
            MatrixToTriplet(ws_->frame_jacobian.topRows<2>(), row_start, col_start, constraint_triplets_, constraint_triplet_idx_);

            // Get configuration linearization
            ws_->frame_jacobian.setZero();
            HolonomicLinearizationq(node, frame, ws_->frame_jacobian);
            col_start = GetDecisionIdx(node, Configuration);
            MatrixToTriplet(ws_->frame_jacobian.topRows<2>(), row_start, col_start, constraint_triplets_, constraint_triplet_idx_);

            // Get the frame vel on the warm start trajectory
            vector3_t frame_vel = robot_model_->GetFrameState(frame, traj_.GetConfiguration(node), traj_.GetVelocity(node)).vel.linear();

            osqp_instance_.lower_bounds.segment<2>(row_start) = -frame_vel.head<2>();
            osqp_instance_.upper_bounds.segment<2>(row_start) = -frame_vel.head<2>();

            row_start+=2;
        }
    }

    // -------------------------------------- //
    // -------- Linearization Helpers ------- //
    // -------------------------------------- //
    matrix3_t FullOrderMpc::QuatIntegrationLinearizationXi(int node) {
        assert(node != nodes_ - 1);
        constexpr double DELTA = 1e-8;

        // TODO: Change for code gen derivative
        quat_t qbar_kp1 = traj_.GetQuat(node+1);
        quat_t qbar_k = traj_.GetQuat(node);
        vector3_t xi = vector3_t::Zero();
        vector3_t w = traj_.GetVelocity(node).segment<3>(POS_VARS);
        const vector3_t xi1_kp1 = robot_model_->QuaternionIntegrationRelative(qbar_kp1, qbar_k, xi, w, dt_[node]);
        matrix3_t update_fd = matrix3_t::Zero();
        for (int i = 0; i < 3; i++) {
            xi(i) += DELTA;
            vector3_t xi2_kp1 = robot_model_->QuaternionIntegrationRelative(qbar_kp1, qbar_k, xi, w, dt_[node]);
            for (int j = 0; j < 3; j++) {
                update_fd(j, i) = (xi2_kp1(j) - xi1_kp1(j))/DELTA;
            }

            xi(i) -= DELTA;
        }

        return update_fd;
    }

    matrix3_t FullOrderMpc::QuatIntegrationLinearizationW(int node) {
        assert(node != nodes_ - 1);

        // TODO: Change for code gen derivative
        quat_t qbar_kp1 = traj_.GetQuat(node+1);
        quat_t qbar_k = traj_.GetQuat(node);
        vector3_t xi = vector3_t::Zero();
        vector3_t w = traj_.GetVelocity(node).segment<3>(POS_VARS);
        const vector3_t xi1_kp1 = robot_model_->QuaternionIntegrationRelative(qbar_kp1, qbar_k, xi, w, dt_[node]);
        matrix3_t update_fd = matrix3_t::Zero();
        for (int i = 0; i < 3; i++) {
            w(i) += FD_DELTA;
            vector3_t xi2_kp1 = robot_model_->QuaternionIntegrationRelative(qbar_kp1, qbar_k, xi, w, dt_[node]);
            for (int j = 0; j < 3; j++) {
                update_fd(j, i) = (xi2_kp1(j) - xi1_kp1(j))/FD_DELTA;
            }

            w(i) -= FD_DELTA;
        }

        return update_fd;
    }

    void FullOrderMpc::InverseDynamicsLinearization(int node, matrixx_t& dtau_dq, matrixx_t& dtau_dv1,
        matrixx_t& dtau_dv2, matrixx_t& dtau_df) {
        assert(node != nodes_ - 1);

        // Compute acceleration via finite difference
        ws_->acc = (traj_.GetVelocity(node + 1) - traj_.GetVelocity(node))/dt_[node];
        // std::cout << "mpc a: " << a.transpose() << std::endl;
        // std::cout << "mpc dt: " << dt_[node] << std::endl;

        // Get external forces
        std::vector<models::ExternalForce> f_ext;
        for (const auto& frame : contact_frames_) {
            f_ext.emplace_back(frame, traj_.GetForce(node, frame));
        }

        robot_model_->InverseDynamicsDerivative(traj_.GetConfiguration(node), traj_.GetVelocity(node),
            ws_->acc, f_ext, dtau_dq, dtau_dv1, dtau_dv2, dtau_df);

        // Note that only the upper triangular part of dtau_da is filled, so we need to extract it
        dtau_dv2.triangularView<Eigen::StrictlyLower>() =
                       dtau_dv2.transpose().triangularView<Eigen::StrictlyLower>();

        // Compute the velocity derivatives
        dtau_dv1 = dtau_dv1 - dtau_dv2/dt_[node];
        dtau_dv2 = dtau_dv2/dt_[node];
    }


    matrix43_t FullOrderMpc::QuatLinearization(int node) {
        matrix43_t q_lin;

        quat_t qbar = traj_.GetQuat(node);
        quat_t q_pert;
        vector3_t pert = vector3_t::Zero();
        for (int i = 0; i < 3; i++) {
            pert(i) += FD_DELTA;
            pinocchio::quaternion::exp3(pert, q_pert);
            pert(i) -= FD_DELTA;
            q_pert = qbar*q_pert;

            q_lin(0, i) = (q_pert.x() - qbar.x())/FD_DELTA;
            q_lin(1, i) = (q_pert.y() - qbar.y())/FD_DELTA;
            q_lin(2, i) = (q_pert.z() - qbar.z())/FD_DELTA;
            q_lin(3, i) = (q_pert.w() - qbar.w())/FD_DELTA;
        }

        return q_lin;
    }

    void FullOrderMpc::SwingHeightLinearization(int node, const std::string& frame, matrix6x_t& jacobian) {
        jacobian.resize(6, robot_model_->GetVelDim());
        jacobian.setZero();
        // *** Note *** The pinocchio body velocity is in the local frame, put I want perturbations to
        // configurations in the world frame, so we can always set the first 3x3 mat to identity
        robot_model_->GetFrameJacobian(frame, traj_.GetConfiguration(node), jacobian );
        jacobian.topLeftCorner<3,3>().setIdentity();
    }

    void FullOrderMpc::HolonomicLinearizationq(int node, const std::string& frame, matrix6x_t& jacobian) {
        robot_model_->FrameVelDerivWrtConfiguration(traj_.GetConfiguration(node),
            traj_.GetVelocity(node), vectorx_t::Zero(robot_model_->GetVelDim()), frame, jacobian);
    }

    void FullOrderMpc::HolonomicLinearizationv(int node, const std::string& frame, matrix6x_t& jacobian) {
        robot_model_->GetFrameJacobian(frame, traj_.GetConfiguration(node), jacobian, pinocchio::WORLD);
    }


    // ------------------------------------------------- //
    // ----------------- Cost Creation ----------------- //
    // ------------------------------------------------- //
    void FullOrderMpc::CreateCostPattern() {
        ws_->obj_config_mat = matrixx_t::Identity(robot_model_->GetVelDim(), robot_model_->GetVelDim());
        ws_->obj_config_mat.block<3, 3>(POS_VARS, POS_VARS) = matrix3_t::Constant(1);

        for (int i = 0; i < config_tracking_weight_.size(); i++) {
            ws_->obj_config_mat.row(i) = ws_->obj_config_mat.row(i)*((config_tracking_weight_(i) != 0) ? 1 : 0);
        }

        ws_->obj_vel_mat = matrixx_t::Identity(robot_model_->GetVelDim(), robot_model_->GetVelDim());
        for (int i = 0; i < vel_tracking_weight_.size(); i++) {
            ws_->obj_vel_mat.row(i) = ws_->obj_vel_mat.row(i)*((vel_tracking_weight_(i) != 0) ? 1 : 0);
        }

        for (int i = 0; i < nodes_; i++) {
            // Configuration Tracking
            int decision_idx = GetDecisionIdx(i, Configuration);
            MatrixToNewTriplet(ws_->obj_config_mat, decision_idx, decision_idx, objective_triplets_);

            // Velocity Tracking
            decision_idx = GetDecisionIdx(i, Velocity);
            MatrixToNewTriplet(ws_->obj_vel_mat, decision_idx, decision_idx, objective_triplets_);
        }

        osqp_instance_.objective_matrix.setFromTriplets(objective_triplets_.begin(), objective_triplets_.end());

        // std::cout << "obj pattern:\n" << osqp_instance_.objective_matrix << std::endl;

        ws_->obj_config_vector.resize(robot_model_->GetVelDim());
        ws_->obj_vel_vector.resize(robot_model_->GetVelDim());
    }

    void FullOrderMpc::UpdateCost() {
        objective_triplet_idx_ = 0;
        osqp_instance_.objective_vector.setZero();

        if (q_target_.size() != v_target_.size()) {
            std::cerr << "Configuration target and velocity target have mis-matched sizes. Filling with default values." << std::endl;
        }

        if (q_target_.size() < nodes_) {
            std::cerr << "Configuration target is missing nodes. Filling with default values." << std::endl;
            for (int i = q_target_.size(); i < nodes_; i++) {
                q_target_.emplace_back(robot_model_->GetNeutralConfig());
            }
        } else if (q_target_.size() > nodes_) {
            std::cerr << "Configuration target has too many nodes. Ignoring extras." << std::endl;
        }


        if (v_target_.size() < nodes_) {
            std::cerr << "Velocity target is missing nodes. Filling with default values." << std::endl;
            for (int i = v_target_.size(); i < nodes_; i++) {
                v_target_.emplace_back(vectorx_t::Zero(robot_model_->GetVelDim()));
            }
        }  else if (v_target_.size() > nodes_) {
            std::cerr << "Velocity target has too many nodes. Ignoring extras." << std::endl;
        }

        for (int node = 0; node < nodes_; node++) {
            int decision_idx = GetDecisionIdx(node, Configuration);
            cost_.Linearize(traj_.GetConfiguration(node), q_target_[node], CostTypes::Configuration, ws_->obj_config_vector);
            osqp_instance_.objective_vector.segment(decision_idx, robot_model_->GetVelDim()) = ws_->obj_config_vector;

            cost_.Quadraticize(traj_.GetConfiguration(node), q_target_[node], CostTypes::Configuration, ws_->obj_config_mat);
            // for (int i = 0; i < 3; i++) {
            //     for (int j = 0; j < 3; j++) {
            //         if (ws_->obj_config_mat(i+3, j+3) == 0 && config_tracking_weight_(i+3) != 0) {
            //             ws_->obj_config_mat(i+3, j+3) += 1e-5; // TODO: Consider changing back to 1e-10
            //         }
            //     }
            // }
            MatrixToTriplet(ws_->obj_config_mat, decision_idx, decision_idx, objective_triplets_, objective_triplet_idx_, true);

            decision_idx = GetDecisionIdx(node, Velocity);
            cost_.Linearize(traj_.GetVelocity(node), v_target_[node], CostTypes::Velocity, ws_->obj_vel_vector);
            osqp_instance_.objective_vector.segment(decision_idx, robot_model_->GetVelDim()) = ws_->obj_vel_vector;

            cost_.Quadraticize(traj_.GetVelocity(node), v_target_[node], CostTypes::Velocity, ws_->obj_vel_mat);
            MatrixToTriplet(ws_->obj_vel_mat, decision_idx, decision_idx, objective_triplets_, objective_triplet_idx_, true);
        }

        objective_mat_.setFromTriplets(objective_triplets_.begin(), objective_triplets_.end());

        // std::cout << "obj mat:\n" << objective_mat_ << std::endl;
        //
        // std::cout << "triplet idx: " << objective_triplet_idx_ << std::endl;
        // std::cout << "triplet size: " << objective_triplets_.size() << std::endl;
        if (objective_triplet_idx_ != objective_triplets_.size()) {
            throw std::runtime_error("[Cost Function] Could not populate the cost function matrix correctly.");
        }
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
            AddTorqueBoxPattern(node);
            if (node > 1) {
                // Configuration is set for the initial condition and the next node, do not constrain it
                AddConfigurationBoxPattern(node);
                AddSwingHeightPattern(node);
            }
            if (node > 0) {
                // Velocity is fixed for the initial condition, do not constrain it
                AddHolonomicPattern(node);
                AddVelocityBoxPattern(node);
            }
        }

        AddFrictionConePattern(nodes_ - 1);
        AddConfigurationBoxPattern(nodes_ - 1);
        AddVelocityBoxPattern(nodes_ - 1);
        AddTorqueBoxPattern(nodes_ - 1);
        AddSwingHeightPattern(nodes_ - 1);
        AddHolonomicPattern(nodes_ - 1);

        int row_max = 0;
        int col_max = 0;
        for (auto constraint_triplet : constraint_triplets_) {
            if (constraint_triplet.row() > row_max) {
                row_max = constraint_triplet.row();
            }
            if (constraint_triplet.col() > col_max) {
                col_max = constraint_triplet.col();
            }
        }

        std::cout << "row max: " << row_max << std::endl;
        std::cout << "col max: " << col_max << std::endl;

        // Make the matrix with the sparsity pattern
        osqp_instance_.constraint_matrix.setFromTriplets(constraint_triplets_.begin(), constraint_triplets_.end());

        // std::cout << "A: \n" << osqp_instance_.constraint_matrix << std::endl;
    }

    void FullOrderMpc::AddICPattern() {
        int row_start = 0;
        int col_start = 0;
        matrixx_t id;
        id.setIdentity(robot_model_->GetVelDim(), robot_model_->GetVelDim());
        MatrixToNewTriplet(id, row_start, col_start, constraint_triplets_);

        row_start += robot_model_->GetVelDim();
        col_start += robot_model_->GetVelDim();

        MatrixToNewTriplet(id, row_start, col_start, constraint_triplets_);
    }

    void FullOrderMpc::AddIntegrationPattern(int node) {
        assert(node != nodes_ - 1);

        const int row_start = GetConstraintRow(node, Integrator);

        // q_k identity, except the quaternion values which are blocks
        int col_start = GetDecisionIdx(node, Configuration);
        ws_->int_mat.setIdentity(robot_model_->GetVelDim(), robot_model_->GetVelDim());
        ws_->int_mat.block<3,3>(POS_VARS, POS_VARS).setConstant(1);
        MatrixToNewTriplet(ws_->int_mat, row_start, col_start, constraint_triplets_);

        // velocity identity except the quaternion values which are blocks
        col_start = GetDecisionIdx(node, Velocity);
        MatrixToNewTriplet(ws_->int_mat, row_start, col_start, constraint_triplets_);

        // q_k+1 negative identity
        col_start = GetDecisionIdx(node + 1, Configuration);
        ws_->int_mat.setIdentity();
        MatrixToNewTriplet(ws_->int_mat, row_start, col_start, constraint_triplets_);
    }

    void FullOrderMpc::AddIDPattern(int node) {
        assert(node != nodes_ - 1);

        const int row_start = GetConstraintRow(node, ID);

        // dtau_dq
        int col_start = GetDecisionIdx(node, Configuration);
        ws_->id_config_mat.setConstant(robot_model_->GetNumInputs() + FLOATING_VEL, robot_model_->GetVelDim(), 1);
        MatrixToNewTriplet(ws_->id_config_mat, row_start, col_start, constraint_triplets_);

        // dtau_dv
        col_start = GetDecisionIdx(node, Velocity);
        ws_->id_vel1_mat.setConstant(robot_model_->GetNumInputs() + FLOATING_VEL, robot_model_->GetVelDim(), 1);
        MatrixToNewTriplet(ws_->id_vel1_mat, row_start, col_start, constraint_triplets_);

        // dtau_dtau
        col_start = GetDecisionIdx(node, Torque);
        matrixx_t id;
        id.setIdentity(robot_model_->GetNumInputs(), robot_model_->GetNumInputs());
        MatrixToNewTriplet(id, row_start + FLOATING_VEL, col_start, constraint_triplets_);

        // dtau_df
        col_start = GetDecisionIdx(node, GroundForce);
        ws_->id_force_mat.setConstant(robot_model_->GetNumInputs() + FLOATING_VEL, num_contact_locations_*CONTACT_3DOF, 1);
        MatrixToNewTriplet(ws_->id_force_mat, row_start, col_start, constraint_triplets_);

        // dtau_dv2
        col_start = GetDecisionIdx(node + 1, Velocity);
        ws_->id_vel2_mat.setConstant(robot_model_->GetNumInputs() + FLOATING_VEL, robot_model_->GetVelDim(), 1);
        MatrixToNewTriplet(ws_->id_vel2_mat, row_start, col_start, constraint_triplets_);
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
        for (int i = 0; i < ws_->fric_cone_mat.rows(); i++) {
            for (int j = 0; j < ws_->fric_cone_mat.cols(); j++) {
                if (ws_->fric_cone_mat(i, j) != 0) {
                    ws_->fric_cone_mat(i, j) = 1;
                }
            }
        }

        for (int contact = 0; contact < num_contact_locations_; contact++) {
            // Setting force to zero when in swing
            MatrixToNewTriplet(id, row_start, col_start, constraint_triplets_);

            row_start += CONTACT_3DOF;

            // Force in friction cone when in contact
            MatrixToNewTriplet(ws_->fric_cone_mat, row_start, col_start, constraint_triplets_);

            row_start += FRICTION_CONE_SIZE;
            col_start += CONTACT_3DOF;
        }
    }

    void FullOrderMpc::AddConfigurationBoxPattern(int node) {
        const int row_start = GetConstraintRow(node, ConfigBox);
        const int col_start = GetDecisionIdx(node, Configuration);

        // all identity except the quaternion elements
        matrixx_t q_box = matrixx_t::Zero(robot_model_->GetConfigDim(), robot_model_->GetConfigDim());
        for (int i = 0; i < q_box.rows(); i++) {
            for (int j = 0; j < q_box.cols(); j++) {
                if (i < 3 && j == i) {
                    q_box(i, j) = 1;
                } else if (i >= FLOATING_BASE && j == i - 1) {
                    q_box(i, j) = 1;
                }
            }
        }
        q_box.block<QUAT_VARS, 3>(POS_VARS, POS_VARS).setConstant(1);

        MatrixToNewTriplet(q_box, row_start, col_start, constraint_triplets_);
    }

    void FullOrderMpc::AddVelocityBoxPattern(int node) {
        const int row_start = GetConstraintRow(node, VelBox);
        const int col_start = GetDecisionIdx(node, Velocity);

        matrixx_t id;
        id.setIdentity(robot_model_->GetVelDim(), robot_model_->GetVelDim());
        MatrixToNewTriplet(id, row_start, col_start, constraint_triplets_);
    }

    void FullOrderMpc::AddTorqueBoxPattern(int node) {
        const int row_start = GetConstraintRow(node, TorqueBox);
        const int col_start = GetDecisionIdx(node, Torque);

        matrixx_t id;
        id.setIdentity(robot_model_->GetNumInputs(), robot_model_->GetNumInputs());
        MatrixToNewTriplet(id, row_start, col_start, constraint_triplets_);
    }

    void FullOrderMpc::AddSwingHeightPattern(int node) {
        int row_start = GetConstraintRow(node, SwingHeight);
        const int col_start = GetDecisionIdx(node, Configuration);

        ws_->swing_vec.setConstant(robot_model_->GetVelDim(), 1);

        for (int contact = 0; contact < num_contact_locations_; contact++) {
            VectorToNewTriplet(ws_->swing_vec, row_start, col_start, constraint_triplets_);
            row_start++;
        }
    }

    void FullOrderMpc::AddHolonomicPattern(int node) {
        int row_start = GetConstraintRow(node, Holonomic);

        ws_->holo_mat.setConstant(2, robot_model_->GetVelDim(), 1);

        for (int contact = 0; contact < num_contact_locations_; contact++) {
            int col_start = GetDecisionIdx(node, Velocity);
            MatrixToNewTriplet(ws_->holo_mat, row_start, col_start, constraint_triplets_);

            col_start = GetDecisionIdx(node, Configuration);
            MatrixToNewTriplet(ws_->holo_mat, row_start, col_start, constraint_triplets_);

            row_start += 2;
        }
    }

    // ------------------------------------------------- //
    // ---------------- Helper Functions --------------- //
    // ------------------------------------------------- //
    void FullOrderMpc::ConvertSolutionToTraj(const vectorx_t& qp_sol, Trajectory& traj) {
        if (traj.GetNumNodes() != nodes_) {
            traj.SetNumNodes(nodes_);
        }

        for (int node = 0; node < nodes_; node++) {
            vectorx_t config_new = traj_.GetConfiguration(node);

            // std::cout << "q dec: " << qp_sol.segment(GetDecisionIdx(node, Configuration), robot_model_->GetVelDim()).transpose()  << std::endl;

            // Position and joints are simple addition
            config_new.head<POS_VARS>() += qp_sol.segment(GetDecisionIdx(node, Configuration), robot_model_->GetVelDim()).head<3>();
            config_new.tail(robot_model_->GetNumInputs()) += qp_sol.segment(GetDecisionIdx(node, Configuration), robot_model_->GetVelDim()).tail(robot_model_->GetNumInputs());

            // Quaternion
            config_new.segment<QUAT_VARS>(POS_VARS) = (static_cast<quat_t>(config_new.segment<QUAT_VARS>(POS_VARS))
                * pinocchio::quaternion::exp3(
                    qp_sol.segment(GetDecisionIdx(node, Configuration), robot_model_->GetVelDim()).segment<3>(POS_VARS))).coeffs();

            traj.SetConfiguration(node, config_new);

            traj.SetVelocity(node,
                qp_sol.segment(GetDecisionIdx(node, Velocity), robot_model_->GetVelDim()) + traj_.GetVelocity(node));
            traj.SetTau(node,
                qp_sol.segment(GetDecisionIdx(node, Torque), robot_model_->GetNumInputs()) + traj_.GetTau(node));
            int force_idx = 0;
            for (const auto& frame : contact_frames_) {
                traj.SetForce(node, frame,
                    qp_sol.segment<3>(GetDecisionIdx(node, GroundForce) + 3*force_idx) + traj_.GetForce(node, frame));
                force_idx++;
            }
        }
    }


    int FullOrderMpc::GetNumConstraints() const {
        // Need to account for the fact that the last node is different
        // The last node does NOT have a ID or integrator constraint
        // The start also has an initial condition constraint
        return GetConstraintsPerNode() * nodes_ - (NumIntegratorConstraintsNode() + NumIDConstraintsNode()
            + 2*NumConfigBoxConstraintsNode() + 2*NumSwingHeightConstraintsNode() + NumHolonomicConstraintsNode() + NumVelocityBoxConstraintsNode())
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
        if (node > 0) {
            row -= NumHolonomicConstraintsNode() + NumVelocityBoxConstraintsNode();
            if (node > 1) {
                row -= 2*NumConfigBoxConstraintsNode() + 2*NumSwingHeightConstraintsNode();
            }
        }

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


    void FullOrderMpc::MatrixToNewTriplet(const torc::mpc::matrixx_t& mat, int row_start, int col_start, std::vector<Eigen::Triplet<double>>& triplet) {
        for (int row = 0; row < mat.rows(); row++) {
            for (int col = 0; col < mat.cols(); col++) {
                // Only in this function do we want to filter out 0's because if they occur here then they are structural
                if (mat(row, col) != 0) {
                    triplet.emplace_back(row_start + row, col_start + col, mat(row, col));
                }
            }
        }
    }

    void FullOrderMpc::VectorToNewTriplet(const vectorx_t& vec, int row_start, int col_start, std::vector<Eigen::Triplet<double>>& triplet) {
        for (int col = 0; col < vec.size(); col++) {
            // Only in this function do we want to filter out 0's because if they occur here then they are structural
            if (vec(col) != 0) {
                triplet.emplace_back(row_start, col_start + col, vec(col));
            }
        }
    }

    void FullOrderMpc::MatrixToTriplet(const torc::mpc::matrixx_t& mat, int row_start, int col_start, std::vector<Eigen::Triplet<double>>& triplet, int&
                                       triplet_idx, bool prune_zeros) {
        for (int row = 0; row < mat.rows(); row++) {
            for (int col = 0; col < mat.cols(); col++) {
                if (!prune_zeros || mat(row, col) != 0) {
                    triplet[triplet_idx] = Eigen::Triplet<double>(row_start + row, col_start + col, mat(row, col));
                    triplet_idx++;
                }
            }
        }
    }

    void FullOrderMpc::VectorToTriplet(const vectorx_t& vec, int row_start, int col_start, std::vector<Eigen::Triplet<double>>& triplet, int& triplet_idx) {
        for (int col = 0; col < vec.size(); col++) {
            triplet[triplet_idx] = Eigen::Triplet<double>(row_start, col_start + col, vec(col));
            triplet_idx++;
        }
    }

    void FullOrderMpc::DiagonalMatrixToTriplet(const torc::mpc::matrixx_t& mat, int row_start, int col_start, std::vector<Eigen::Triplet<double>>& triplet, int&
                                               triplet_idx) {
        for (int idx = 0; idx < mat.rows(); idx++) {
            // Don't filter zero's here as they aren't structural
            triplet[triplet_idx] = Eigen::Triplet<double>(row_start + idx, col_start + idx, mat(idx, idx));
            triplet_idx++;
        }
    }

    void FullOrderMpc::DiagonalScalarMatrixToTriplet(double val, int row_start, int col_start, int size, std::vector<Eigen::Triplet<double>>& triplet, int& triplet_idx) {
        for (int idx = 0; idx < size; idx++) {
            triplet[triplet_idx] = Eigen::Triplet<double>(row_start + idx, col_start + idx, val);
            triplet_idx++;
        }
    }

    // ------------------------------------------------- //
    // ----- Getters for Sizes of Individual nodes ----- //
    // ------------------------------------------------- //
    int FullOrderMpc::NumIntegratorConstraintsNode() const {
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
        return robot_model_->GetConfigDim();
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

    std::vector<std::string> FullOrderMpc::GetContactFrames() const {
        return contact_frames_;
    }

    int FullOrderMpc::GetNumNodes() const {
        return nodes_;
    }

    void FullOrderMpc::SetWarmStartTrajectory(const Trajectory& traj) {
        traj_ = traj;
    }

    void FullOrderMpc::PrintStatistics() const {
        using std::setw;
        using std::setfill;

        const int col_width = 20;
        const int total_width = 7*col_width;


        auto time_now = std::chrono::system_clock::now();
        std::time_t time1_now = std::chrono::system_clock::to_time_t(time_now);

        std::cout << setfill('=') << setw(total_width/2 - 7) << "" << " MPC Statistics " << setw(total_width/2 - 7) << "" << std::endl;
        std::cout << setfill(' ');
        std::cout << setw(col_width) << "Solve #"
                << setw(col_width) << "Solve Status"
                << setw(col_width) << "Time (ms)"
                << setw(col_width) << "Step Norm"
                << setw(col_width) << "Alpha"
                << setw(col_width) << "Full Cost"
                << setw(col_width) << "QP Cost" << std::endl;
        // << setw(col_width) << "Constraints"
        // << setw(col_width) << "Merit"
        // << setw(col_width) << "Merit dd"
        for (int solve = 0; solve < stats_.size(); solve++) {
            std::string solve_status;
            switch (stats_[solve].solve_status) {
                case osqp::OsqpExitCode::kOptimal:
                    solve_status = "Optimal Sol. Found";
                    break;
                case osqp::OsqpExitCode::kPrimalInfeasible:
                    solve_status = "Primal Infeasible";
                    break;
                case osqp::OsqpExitCode::kDualInfeasible:
                    solve_status = "Dual Infeasible";
                    break;
                case osqp::OsqpExitCode::kOptimalInaccurate:
                    solve_status = "Optimal Inaccurate";
                    break;
                case osqp::OsqpExitCode::kPrimalInfeasibleInaccurate:
                    solve_status = "Primal Infeasible Inaccurate";
                    break;
                case osqp::OsqpExitCode::kDualInfeasibleInaccurate:
                    solve_status = "Dual Infeasible Inaccurate";
                    break;
                case osqp::OsqpExitCode::kMaxIterations:
                    solve_status = "Max Iterations";
                    break;
                case osqp::OsqpExitCode::kInterrupted:
                    solve_status = "Interrupted";
                    break;
                case osqp::OsqpExitCode::kNonConvex:
                    solve_status = "Non-Convex";
                    break;
                case osqp::OsqpExitCode::kTimeLimitReached:
                    solve_status = "Time Limit";
                    break;
                case osqp::OsqpExitCode::kUnknown:
                    solve_status = "Unknown";
                    break;
            }
            std::cout << setw(col_width) << solve
                      << setw(col_width) << solve_status
                      << setw(col_width) << stats_[solve].total_compute_time
                      << setw(col_width) << stats_[solve].qp_res_norm
                      << setw(col_width) << stats_[solve].alpha
                      << setw(col_width) << stats_[solve].full_cost
                      << setw(col_width) << stats_[solve].qp_cost << std::endl;
        }


    }
} // namespace torc::mpc
