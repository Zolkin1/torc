//
// Created by zolkin on 7/28/24.
//
#include <iostream>

#include "full_order_mpc.h"
#include "autodiff_fn.h"

#include <pinocchio/algorithm/kinematics-derivatives.hxx>
#include <Eigen/Dense>
#include <mutex>

#include "eigen_utils.h"
#include "torc_timer.h"
#include "pinocchio_interface.h"

#define NAN_CHECKS 1

//  Can make the max force different for each contact, so I make sure the toe force is small (i.e. proportional to the ankle torque)
// TODO: Consider removing the last torque and force variables (i.e. in the last node) - would make the problem slightly smaller
// TODO: If I compute the linearization about the trajectory (except the first node) before I recieve the state estimate
//  then I reduce the time delay between recieving the estimate and the outputting a control. (Total computation time
//  is the same).

namespace torc::mpc {
    FullOrderMpc::FullOrderMpc(const std::string& name, const fs::path& config_file, const fs::path& model_path)
        : config_file_(config_file), verbose_(true), name_(name), cost_(name), compile_derivatves_(true), scale_cost_(false),
            total_solves_(0), enable_delay_prediction_(true) {
        // Verify the robot file exists
        if (!fs::exists(model_path)) {
            throw std::runtime_error("Robot file does not exist!");
        }
        robot_model_ = std::make_unique<models::FullOrderRigidBody>("mpc_robot", model_path);

        vel_dim_ = robot_model_->GetVelDim();
        config_dim_ = robot_model_->GetConfigDim();
        input_dim_ = robot_model_->GetNumInputs();

        // Verify the config file exists
        if (!fs::exists(config_file_)) {
            throw std::runtime_error("Configuration file does not exist!");
        }
        UpdateSettings();

        ws_ = std::make_unique<Workspace>();
        traj_.UpdateSizes(robot_model_->GetConfigDim(), robot_model_->GetVelDim(),
                          robot_model_->GetNumInputs(), contact_frames_, nodes_);
        traj_.SetDtVector(dt_);

        // ---------- Create the ad functions ---------- //
        integration_constraint_ = std::make_unique<ad::CppADInterface>(
            std::bind(&FullOrderMpc::IntegrationConstraint, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
            name_ + "_integration_constraint",
            deriv_lib_path_,
            ad::DerivativeOrder::FirstOrder, 4*vel_dim_, 1 + 2*config_dim_ + 2*vel_dim_,
            compile_derivatves_
        );

        for (const auto& frame : contact_frames_) {
            holonomic_constraint_.emplace(frame,
                std::make_unique<ad::CppADInterface>(
                    std::bind(&FullOrderMpc::HolonomicConstraint, this, frame, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                    name_ + "_" + frame + "_holonomic_constraint",
                    deriv_lib_path_,
                    ad::DerivativeOrder::FirstOrder, 2*vel_dim_,  config_dim_ + vel_dim_,
                    compile_derivatves_
                ));

            swing_height_constraint_.emplace(frame,
            std::make_unique<ad::CppADInterface>(
                std::bind(&FullOrderMpc::SwingHeightConstraint, this, frame, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
               name_ + "_" + frame + "_swing_height_constraint",
               deriv_lib_path_,
               ad::DerivativeOrder::FirstOrder, vel_dim_,  config_dim_ + 1,
               compile_derivatves_
            ));
        }

        inverse_dynamics_constraint_ = std::make_unique<ad::CppADInterface>(
            std::bind(&FullOrderMpc::InverseDynamicsConstraint, this, contact_frames_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
            name_ + "_inverse_dynamics_constraint",
            deriv_lib_path_,
            ad::DerivativeOrder::FirstOrder, 3*vel_dim_ + input_dim_ + CONTACT_3DOF*num_contact_locations_,
            1 + config_dim_ + 2*vel_dim_ + input_dim_ + CONTACT_3DOF*num_contact_locations_,
            compile_derivatves_
        );

        friction_cone_constraint_ = std::make_unique<ad::CppADInterface>(
            std::bind(&FullOrderMpc::FrictionConeConstraint, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
            name_ + "_friction_cone_constraint",
            deriv_lib_path_,
            ad::DerivativeOrder::FirstOrder, 3, 4,
            compile_derivatves_
        );

    }

    FullOrderMpc::~FullOrderMpc() {
        if (verbose_) {
            PrintAggregateStats();
        }
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
                if (nodes_ < 3) {
                    throw std::invalid_argument("nodes must be >= 3!");
                }
            } else {
                throw std::runtime_error("Number of nodes not specified!");
            }

            if (general_settings["verbose"]) {
                verbose_ = general_settings["verbose"].as<bool>();
            }

            int node_type = 0;
            if (general_settings["node_dt_type"]) {
                if (general_settings["node_dt_type"].as<std::string>() == "SmallFirst") {
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
                    const auto dt = general_settings["node_dt"].as<double>();
                    dt_.resize(nodes_);
                    for (double & it : dt_) {
                        it = dt;
                    }
                } else {
                    throw std::runtime_error("Node dt not specified!");
                }
            } else if (node_type == 1) {
                if (general_settings["node_dt"]) {
                    const auto dt = general_settings["node_dt"].as<double>();
                    dt_.resize(nodes_);
                    for (double & it : dt_) {
                        it = dt;
                    }
                } else {
                    throw std::runtime_error("Node dt not specified!");
                }
                if (general_settings["first_node_dt"]) {
                    dt_[0] = general_settings["first_node_dt"].as<double>();
                } else {
                    throw std::runtime_error("You selected node dt type as 'SmallFirst' but did not provide a first node dt!");
                }
            }

            if (general_settings["compile_derivatives"]) {
                compile_derivatves_ = general_settings["compile_derivatives"].as<bool>();
            }

            if (general_settings["deriv_lib_path"]) {
                deriv_lib_path_ = general_settings["deriv_lib_path"].as<std::string>();
            } else {
                deriv_lib_path_ = fs::current_path();
                deriv_lib_path_ = deriv_lib_path_ / "deriv_libs";
            }

            if (general_settings["base_frame"]) {
                base_frame_ = general_settings["base_frame"].as<std::string>();
            } else {
                throw std::runtime_error("No base frame name provided in configuration file!");
            }

            if (general_settings["scale_cost"]) {
                scale_cost_ = general_settings["scale_cost"].as<bool>();
            } else {
                scale_cost_ = false;
            }

            if (general_settings["max_initial_solves"]) {
                max_initial_solves_ = general_settings["max_initial_solves"].as<int>();
            } else {
                max_initial_solves_ = 10;
            }

            if (general_settings["initial_constraint_tol"]) {
                initial_constraint_tol_ = general_settings["initial_constraint_tol"].as<double>();
            } else {
                initial_constraint_tol_ = 1e-2;
            }

            if (general_settings["nodes_full_dynamics"]) {
                nodes_full_dynamics_ = general_settings["nodes_full_dynamics"].as<int>();
                if (nodes_full_dynamics_ > nodes_ - 1) {
                    throw std::invalid_argument("The nodes with full dynamics must be <= total nodes - 1");
                }
            } else {
                nodes_full_dynamics_ = std::min(5, nodes_ - 1);
            }

            terminal_cost_weight_ = (general_settings["terminal_cost_weight"] ? general_settings["terminal_cost_weight"].as<double>() : 1.0);

            delay_prediction_dt_ = (general_settings["delay_prediction_dt"] ? general_settings["delay_prediction_dt"].as<double>() : 0);
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
            if (solver_settings["scaling"]) {
                osqp_settings_.scaling = solver_settings["scaling"].as<int>();
            }
            if (solver_settings["sigma"]) {
                osqp_settings_.sigma = solver_settings["sigma"].as<double>();
            }
        }

        // ---------- Constraint Settings ---------- //
        if (!config["constraints"]) {
            throw std::runtime_error("No constraint settings provided!");
        }
        YAML::Node constraint_settings = config["constraints"];
        friction_coef_ = constraint_settings["friction_coef"].as<double>();
        max_grf_ = constraint_settings["max_grf"].as<double>();
        friction_margin_ = (constraint_settings["friction_margin"] ? constraint_settings["friction_margin"].as<double>() : 0.05);

        ParseCollisionYaml(constraint_settings["collisions"]);

        // ---------- Cost Settings ---------- //
        if (!config["costs"]) {
            throw std::runtime_error("No cost settings provided!");
        }
        YAML::Node cost_settings = config["costs"];

        ParseCostYaml(cost_settings);

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

            // Set everything to not in contact
            in_contact_[frame] = std::vector<int>(nodes_);
        }

        if (contact_settings["hip_offsets"]) {
            hip_offsets_ = contact_settings["hip_offsets"].as<std::vector<double>>();
            if (hip_offsets_.size() != 2*contact_frames_.size()) {
                throw std::runtime_error("Invalid number of hip offsets provided! Must match the number of contacts x 2!");
            }
        }

        // ---------- Line Search Settings ---------- //
        if (!config["line_search"]) {
            throw std::runtime_error("No line search setting provided!");
        } else {
            YAML::Node ls_settings = config["line_search"];
            ls_eta_ = (ls_settings["armijo_constant"] ? ls_settings["armijo_constant"].as<double>() : 1e-4);
            ls_alpha_min_ = (ls_settings["alpha_min"] ? ls_settings["alpha_min"].as<double>() : 1e-4);
            ls_theta_max_ = (ls_settings["large_constraint_vio"] ? ls_settings["large_constraint_vio"].as<double>() : 1e-2);
            ls_theta_min_ = (ls_settings["small_constraint_vio"] ? ls_settings["small_constraint_vio"].as<double>() : 1e-6);
            ls_gamma_theta_ = (ls_settings["constraint_reduction_mult"] ? ls_settings["constraint_reduction_mult"].as<double>() : 1e-6);
            ls_gamma_alpha_ = (ls_settings["alpha_step"] ? ls_settings["alpha_step"].as<double>() : 0.5);
            ls_gamma_phi_ = (ls_settings["cost_reduction_mult"] ? ls_settings["cost_reduction_mult"].as<double>() : 1e-6);
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
            std::cout << "\tDerivative library location: " << deriv_lib_path_.string() << std::endl;
            std::cout << "\tScale cost: " << (scale_cost_ ? "True" : "False") << std::endl;
            std::cout << "\tMax initial solves: " << max_initial_solves_ << std::endl;
            std::cout << "\tInitial constraint tolerance: " << initial_constraint_tol_ << std::endl;
            std::cout << "\tNodes with full dynamics: " << nodes_full_dynamics_ << std::endl;
            std::cout << "\tDelay prediction dt: " << (delay_prediction_dt_ < 0 ? "Adaptive" : std::to_string(delay_prediction_dt_)) << std::endl;

            std::cout << "Solver settings: " << std::endl;
            std::cout << "\tRelative tolerance: " << osqp_settings_.eps_rel << std::endl;
            std::cout << "\tAbsolute tolerance: " << osqp_settings_.eps_abs << std::endl;
            std::cout << "\tVerbose: " << (osqp_settings_.verbose ? "True" : "False") << std::endl;
            std::cout << "\tPolish: " << (osqp_settings_.polish ? "True" : "False") << std::endl;
            std::cout << "\trho: " << osqp_settings_.rho << std::endl;
            std::cout << "\talpha: " << osqp_settings_.alpha << std::endl;
            std::cout << "\tsigma: " << osqp_settings_.sigma << std::endl;
            std::cout << "\tAdaptive rho: " << (osqp_settings_.adaptive_rho ? "True" : "False") << std::endl;
            std::cout << "\tMax iterations: " << osqp_settings_.max_iter << std::endl;
            std::cout << "\tScaling: " << osqp_settings_.scaling << std::endl;

            std::cout << "Constraints:" << std::endl;
            std::cout << "\tFriction coefficient: " << friction_coef_ << std::endl;
            std::cout << "\tFriction margin: " << friction_margin_ << std::endl;
            std::cout << "\tMaximum ground reaction force: " << max_grf_ << std::endl;

            std::cout << "Costs:" << std::endl;
            for (const auto& data : cost_data_) {
                std::cout << "\tCost name: " << data.constraint_name << std::endl;
                std::cout << "\t\tWeight: " << data.weight.transpose() << std::endl;
                if (data.frame_name != "") {
                    std::cout << "\t\tFrame name: " << data.frame_name << std::endl;
                }
            }
            std::cout << "\tTerminal cost weight: " << terminal_cost_weight_ << std::endl;

            std::cout << "Contacts:" << std::endl;
            std::cout << "\tNumber of contact locations: " << num_contact_locations_ << std::endl;
            std::cout << "\tContact frames: [ ";
            for (const auto& frame : contact_frames_) {
                std::cout << frame << " ";
            }
            std::cout << "]" << std::endl;

            std::cout << "Line search:" << std::endl;
            std::cout << "\tArmijo constant: " << ls_eta_ << std::endl;
            std::cout << "\tLarge constraint violation: " << ls_theta_max_ << std::endl;
            std::cout << "\tSmall constraint violation: " << ls_theta_min_ << std::endl;
            std::cout << "\tConstraint reduction multiplier: " << ls_gamma_theta_ << std::endl;
            std::cout << "\tCost reduction multiplier: " << ls_gamma_phi_ << std::endl;
            std::cout << "\tLine search step: " << ls_gamma_alpha_ << std::endl;
            std::cout << "\tSmallest step: " << ls_alpha_min_ << std::endl;

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

        total_solves_ = 0;

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
        traj_.SetDtVector(dt_);

        // Setup remaining workspace memory
        ws_->acc.resize(robot_model_->GetVelDim());
        for (const auto& frame : contact_frames_) {
            ws_->f_ext.emplace_back(frame, vector3_t::Zero());
        }
        ws_->frame_jacobian.resize(6, robot_model_->GetVelDim());

        // Setup cost function
        cost_.Configure(robot_model_, compile_derivatves_, cost_data_, deriv_lib_path_);

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

    void FullOrderMpc::UpdateContactSchedule(const ContactSchedule& contact_schedule) {
        // Convert the contact schedule into the binary digits I need
        for (auto& [frame, schedule] : contact_schedule.GetScheduleMap()) {
            if (in_contact_.contains(frame)) {
                // Break the continuous time contact schedule into each node
                double time = 0;
                for (int node = 0; node < nodes_; node++) {
                    if (contact_schedule.InContact(frame, time)) {
                        in_contact_[frame][node] = 1;
                    } else {
                        in_contact_[frame][node] = 0;
                    }
                    time += dt_[node];
                }
            } else {
                throw std::runtime_error("Contact schedule contains contact frames not recognized by the MPC!");
            }
        }

    }

    void FullOrderMpc::UpdateContactScheduleAndSwingTraj(const ContactSchedule& contact_schedule, double apex_height,
            std::vector<double> end_height, double apex_time) {
        UpdateContactSchedule(contact_schedule);

        if (end_height.size() != num_contact_locations_) {
            throw std::runtime_error("end_height size does not match the number of contacts in the MPC!");
        }

        int frame_idx = 0;
        for (auto& [frame, traj] : swing_traj_) {
            contact_schedule.CreateSwingTraj(frame, apex_height, end_height[frame_idx], apex_time, dt_, traj);
            frame_idx++;
        }
    }

    void FullOrderMpc::Compute(const vectorx_t& q, const vectorx_t& v, Trajectory& traj_out, double delay_start_time) {
        utils::TORCTimer compute_timer;
        compute_timer.Tic();

        // Delay prediction
        vectorx_t q_new = q;
        vectorx_t v_new = v;

        if (delay_prediction_dt_ != 0 && enable_delay_prediction_) {
            // TODO: Fix!
            throw std::runtime_error("Delay prediction not implemented yet!");

            double total_dt;
            if (delay_prediction_dt_ < 0) {
                // Average the last 5 computation times
                if (stats_.size() >= 5) {
                    total_dt = 0;
                    for (int i = 0; i < 5; i++) {
                        total_dt += stats_[stats_.size() - (i + 1)].total_compute_time;
                    }
                    total_dt = total_dt / 5e3;
                } else {
                    total_dt = 1e-2;
                }
            } else {
                total_dt = delay_prediction_dt_;
            }

            const int steps = 2;
            double dt = total_dt / static_cast<double>(steps);

            vectorx_t tau;
            for (int i = 0; i < steps; i++) {
                double curr_time = i * dt;
                traj_.GetTorqueInterp(delay_start_time + curr_time, tau);
                for (auto& f : ws_->f_ext) {
                    traj_.GetForceInterp(delay_start_time + curr_time, f.frame_name, f.force_linear);
                }

                vectorx_t xdot = robot_model_->GetDynamics(q_new, v_new, tau, ws_->f_ext);
                q_new = robot_model_->IntegrateVelocity(q_new, dt*xdot.head(v.size()));
                v_new = v_new + dt*xdot.tail(v.size());
            }

            // TODO: Measure the error between this and the measured one

//            std::cout << "q delay: " << q_new.transpose() << std::endl;
//            std::cout << "v delay: " << v_new.transpose() << std::endl;
        }

//        std::cout << "Current traj config node 0: " << traj_.GetConfiguration(0).transpose() << std::endl;
//        std::cout << "Current traj vel node 0: " << traj_.GetVelocity(0).transpose() << std::endl;

        // Normalize quaternion to be safe
        q_new.segment<QUAT_VARS>(POS_VARS).normalize();

        traj_.SetConfiguration(0, q_new);
        traj_.SetVelocity(0, v_new);
        utils::TORCTimer constraint_timer;
        constraint_timer.Tic();
        CreateConstraints();
        constraint_timer.Toc();

#if NAN_CHECKS
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
#endif

         utils::TORCTimer cost_timer;
         cost_timer.Tic();
         UpdateCost();
         cost_timer.Toc();
#if NAN_CHECKS
        for (const auto& objective_triplet : objective_triplets_) {
            if(std::isnan(objective_triplet.value())) {
                throw std::runtime_error("nan in constraint mat");
            }

            // if (objective_triplet.value() == 0) {
                // std::cerr << "objective 0 in triplet!" << std::endl;
                // throw std::runtime_error("objective 0 in triplet!");
            // }
        }

        for (const auto& obj_vec : osqp_instance_.objective_vector) {
            if (std::isnan(obj_vec)) {
                throw std::runtime_error("nan in objective vector");
            }
        }
#endif

         // std::cout << "objective mat: \n" << objective_mat_ << std::endl;
         // std::cout << "objective vec: " << osqp_instance_.objective_vector.transpose() << std::endl;
         // std::cout << "A: \n" << A_ << std::endl;

        // Set upper and lower bounds
        auto status = osqp_solver_.SetBounds(osqp_instance_.lower_bounds, osqp_instance_.upper_bounds);
        if (!status.ok()) {
            std::cerr << "status: " << status << std::endl;
            throw std::runtime_error("Could not update the constraint bounds.");
        }

         // Set matrices
        status = osqp_solver_.UpdateObjectiveAndConstraintMatrices(objective_mat_, A_);
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

        // Set the warmstart to 0 -- appears to be very important
        status = osqp_solver_.SetWarmStart(vectorx_t::Zero(GetNumDecisionVars()),
            vectorx_t::Zero(GetNumConstraints()));
        if (!status.ok()) {
            std::cerr << "status: " << status << std::endl;
            throw std::runtime_error("Could not update the warmstart.");
        }

        // Solve
        auto solve_status = osqp_solver_.Solve();

        if (solve_status == osqp::OsqpExitCode::kPrimalInfeasible || solve_status == osqp::OsqpExitCode::kPrimalInfeasibleInaccurate) {
            std::cerr << "Bad solve!" << std::endl;
            throw std::runtime_error("Infeasible!");
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

        utils::TORCTimer ls_timer;
        ls_timer.Tic();
        // TODO: Keep an eye on this
        // auto [constraint_ls, cost_ls] = LineSearch(osqp_solver_.primal_solution());
        double cost_ls = GetFullCost(osqp_solver_.primal_solution());
        double constraint_ls = GetConstraintViolation(osqp_solver_.primal_solution());  // TODO: Fix this function!
        alpha_ = 1;
        ls_timer.Toc();

        // for (const auto& frame : contact_frames_) {
        //     int node = 1;
        //     vectorx_t x(swing_height_constraint_[frame]->GetDomainSize());
        //     x << osqp_solver_.primal_solution().segment(GetDecisionIdx(node, Configuration), vel_dim_);
        //     vectorx_t p(swing_height_constraint_[frame]->GetParameterSize());
        //     p << traj_.GetConfiguration(node), swing_traj_[frame][node];
        //     vectorx_t y;
        //     swing_height_constraint_[frame]->GetFunctionValue(x, p, y);
        //     std::cout << "swing height error for frame " << frame << " is " << y << std::endl;
        //     std::cout << "desired height: " << swing_traj_[frame][node] << std::endl << std::endl;
        // }

#if NAN_CHECKS
        if (std::isnan(constraint_ls)) {
            throw std::runtime_error("nan in constraint violation line search");
        }

        if (std::isnan(cost_ls)) {
            throw std::runtime_error("nan in cost line search");
        }
#endif

        ConvertSolutionToTraj(alpha_*osqp_solver_.primal_solution(), traj_out);
        traj_out.SetDtVector(dt_);
         // std::cout << "solve result: \n" << osqp_solver_.primal_solution() << std::endl;

        compute_timer.Toc();


         stats_.emplace_back(solve_status, osqp_solver_.objective_value(),
             cost_ls, alpha_, (alpha_*osqp_solver_.primal_solution()).norm(),
             compute_timer.Duration<std::chrono::microseconds>().count()/1000.0,
             constraint_timer.Duration<std::chrono::microseconds>().count()/1000.0,
             cost_timer.Duration<std::chrono::microseconds>().count()/1000.0,
             ls_timer.Duration<std::chrono::microseconds>().count()/1000.0,
             ls_condition_, constraint_ls);

        traj_ = traj_out;

        // double post_cost = GetFullCost(vectorx_t::Zero(GetNumDecisionVars()));
        // double post_constraint = GetConstraintViolation(vectorx_t::Zero(GetNumDecisionVars()));
        // if (std::abs(cost_ls - post_cost) > 1e-8) {
        //     std::cerr << "Cost Difference: " << std::abs(cost_ls - post_cost) << std::endl;
        //     // throw std::runtime_error("Traj conversion effected cost!");
        // }
        //
        // if (std::abs(constraint_ls - post_constraint) > 1e-8) {
        //     std::cerr << "Constraint Difference: " << std::abs(constraint_ls - post_constraint) << std::endl;
        //     // throw std::runtime_error("Traj conversion effected constraint violation!");
        // }

        total_solves_++;

        if (verbose_) {
             std::cout << "MPC compute took " << compute_timer.Duration<std::chrono::microseconds>().count()/1000.0 << "ms." << std::endl;
        }
    }

    void FullOrderMpc::ComputeNLP(const vectorx_t& q, const vectorx_t& v, Trajectory& traj_out) {
        // No delay compensation here as we are assumed to not be moving.
        enable_delay_prediction_ = false;
        Compute(q, v, traj_out);
        for (int i = 0; i < max_initial_solves_; i++) {
            if (stats_[i].constraint_violation < initial_constraint_tol_) {
                std::cout << "Initial compute constraint violation converged in " << i+1 << " QP solves." << std::endl;
                return;
            }
            Compute(q, v, traj_out);
        }

        std::cerr << "Did not reach constraint tolerance of " << initial_constraint_tol_ << " after " << max_initial_solves_ << " QP solves." << std::endl;

        if (verbose_) {
            PrintStatistics();
        }

        enable_delay_prediction_ = true;
    }

    void FullOrderMpc::ShiftWarmStart(double dt) {
        // Update the warm start trajectory
        // Shift all but the first node
        vectorx_t q, v, tau;
        vector3_t force;
        for (int node = 1; node < nodes_; node++) {
            const double time = GetTime(node) + dt;

            traj_.GetConfigInterp(time, q);
            traj_.SetConfiguration(node, q);

            traj_.GetVelocityInterp(time, v);
            traj_.SetVelocity(node, v);

            traj_.GetTorqueInterp(time, tau);
            traj_.SetTau(node, tau);

            for (const auto& frame : contact_frames_) {
                traj_.GetForceInterp(time, frame, force);
                traj_.SetForce(node, frame, force);
            }
        }
    }

    void FullOrderMpc::GenerateCostReference(const vectorx_t& q, const vectorx_t& v, const vector3_t& vel) {
        if (hip_offsets_.empty()) {
            throw std::runtime_error("No hip joint provided in the config file! Cannot generate the cost reference!");
        }

        for (int node = 0; node < nodes_; node++) {
            // Integrate the velocity forward to get the planar positions
            q_target_.value()[node](0) = node*dt_[node]*vel(0) + q(0);
            q_target_.value()[node](1) = node*dt_[node]*vel(1) + q(1);
            q_target_.value()[node](2) = q(2);
            // TODO: Determine the quaternion targets from the commanded velocity, for now ignoring
            q_target_.value()[node].segment<3>(POS_VARS).setZero();
            q_target_.value()[node](6) = 1;

            // Assign velocity targets
            v_target_.value()[node](0) = vel(0);
            v_target_.value()[node](1) = vel(1);
            v_target_.value()[node](2) = 0;
            // TODO: Determine the quaternion velocity from the commanded velocity, for now setting to 0
            v_target_.value()[node].segment<3>(POS_VARS).setZero();
            v_target_.value()[node].tail(robot_model_->GetVelDim() - FLOATING_VEL).setZero();
        }

        // Compute foothold positions
        std::map<std::string, std::vector<vector3_t>> positions;
        int frame_idx = 0;
        for (const auto& frame : contact_frames_) {
            positions[frame] = std::vector<vector3_t>(nodes_);

            vector2_t foothold = vector2_t::Zero();
            bool first_contact = false;

            for (int node = 0; node < nodes_; node++) {
                if (in_contact_[frame][node]) {
                    if (node == 0 && in_contact_[frame][node]) {
                        // Foothold is where we currently are
                        robot_model_->FirstOrderFK(q);
                        foothold = robot_model_->GetFrameState(frame).placement.translation().head<2>();
                        first_contact = true;
                    } else if (node > 0 && (!in_contact_[frame][node-1] && in_contact_[frame][node])) {
                        // Compute the middle node of the contact
                        int contact_count = 0;
                        while (contact_count < nodes_ && in_contact_[frame][node + contact_count]) {
                            contact_count++;
                        }

                        // Flooring division, could be ceil
                        int contact_middle = std::floor(contact_count / 2);
                        int contact_node_middle = node + contact_middle;

                        // Compute x,y reference given this node
                        vector2_t hip_offset;
                        hip_offset << hip_offsets_[2*frame_idx], hip_offsets_[2*frame_idx + 1];
                        // std::cout << "frame: " << frame << ", hip offset: " << hip_offset << std::endl;

                        foothold = hip_offset + q_target_.value()[contact_node_middle].head<2>();

                        if (!first_contact) {
                            first_contact = true;
                            vector2_t raibert_offset;
                            raibert_offset << std::sqrt(q(2)/9.81)*(v(0) - vel(0)),
                                              std::sqrt(q(2)/9.81)*(v(1) - vel(1));
                            foothold = foothold + raibert_offset;
                        }

                        // std::cout << "frame: " << frame << ", foothold: " << foothold.transpose() << std::endl;
                    }

                    positions[frame][node] << foothold(0), foothold(1), swing_traj_[frame][node];
                }
            }
            frame_idx++;
        }


        // std::vector<std::string> frames;
        // for (int i = 0; i < contact_frames_.size(); i++) {
        //     if (hip_offsets_[i] != "none") {
        //         frames.push_back(contact_frames_[i]);
        //     }
        // }

        robot_model_->FirstOrderFK(q);
        frame_idx = 0;
        for (const auto& frame : contact_frames_) {
            vector2_t prev_foothold = robot_model_->GetFrameState(frame).placement.translation().head<2>();
            vector2_t next_foothold = robot_model_->GetFrameState(frame).placement.translation().head<2>();
            // std::cout << "Frame: " << frame << std::endl;
            int swing_start = 0;
            int swing_end = 0;
            for (int node = 0; node < nodes_; node++) {
                // Linear interpolation between footholds
                if ((node == 0 && !in_contact_[frame][node]) || (node > 0 && (!in_contact_[frame][node] && in_contact_[frame][node-1]))) {
                    swing_start = node;
                }

                if (swing_start == node) {
                    swing_end = swing_start;
                    while (swing_end < nodes_ && !in_contact_[frame][swing_end]) {
                        swing_end++;
                    }

                    if (swing_end == nodes_) {
                        swing_end += 5; // TODO: Deal with this better, for now just extending it a bit
                        // TODO: Consider changing how this next foothold is computed
                        next_foothold = prev_foothold + dt_[1]*(swing_end - swing_start)*vel.head<2>();
                    } else {
                        next_foothold = positions[frame][swing_end].head<2>();
                    }
                }

                if (!in_contact_[frame][node]) {
                    vector2_t swing_location = prev_foothold + (next_foothold - prev_foothold)*(static_cast<double>(node - swing_start)/static_cast<double>(swing_end - swing_start));
                    // std::cout << "node: " << node << ", swing_start: " << swing_start << ", swing end: " << swing_end << std::endl;
                    // std::cout << "prev_foothold: " << prev_foothold.transpose() << std::endl;
                    // std::cout << "next_foothold: " << next_foothold.transpose() << std::endl;
                    // std::cout << "swing location: " << swing_location.transpose() << std::endl;
                    positions[frame][node] << swing_location(0), swing_location(1), swing_traj_[frame][node];
                } else {
                    prev_foothold = positions[frame][node].head<2>();
                }

                // std::cout << node << ": " << positions[frame][node].transpose() << std::endl;
                // std::cout << "start swing: " << swing_start << ", end swing: " << swing_end << std::endl;
                // std::cout << "in contact: " << in_contact_[frame][node] << std::endl;
            }
            frame_idx++;
        }

        // TODO: Do I want this?
        q_target_.value()[0] = q;
        for (int node = 1; node < nodes_; node++) {
            std::vector<vector3_t> end_effectors(contact_frames_.size());
            for (int i = 0; i < contact_frames_.size(); i++) {
                end_effectors[i] = positions[contact_frames_[i]][node];
                // std::cout << "node: " << node << ", target: " << positions[contact_frames_[i]][node].transpose() << std::endl;
            }
            // IK for joint targets
            // q_target_.value()[node].tail(robot_model_->GetConfigDim() - FLOATING_BASE) =
            //     robot_model_->InverseKinematics(q_target_.value()[node].head<FLOATING_BASE>(),
            //         end_effectors, frames, traj_.GetConfiguration(node)).tail(robot_model_->GetConfigDim() - FLOATING_BASE);


            q_target_.value()[node].tail(robot_model_->GetConfigDim() - FLOATING_BASE) =
                robot_model_->InverseKinematics(q_target_.value()[node].head<FLOATING_BASE>(),
                    end_effectors, contact_frames_, q_target_.value()[node-1]).tail(robot_model_->GetConfigDim() - FLOATING_BASE);
        }

    }

    SimpleTrajectory FullOrderMpc::GetConfigTargets() {
        return q_target_.value();
    }



    // ------------------------------------------------- //
    // -------------- Constraint Creation -------------- //
    // ------------------------------------------------- //
    void FullOrderMpc::CreateConstraints() {
        constraint_triplet_idx_ = 0;
        osqp_instance_.lower_bounds.setZero();
        osqp_instance_.upper_bounds.setZero();

        for (int node = 0; node < nodes_; node++) {
            // Dynamics related constraints don't happen in the last node
            if (node < nodes_ - 1) {
                AddIntegrationConstraint(node);
            }

            if (node < nodes_full_dynamics_) {
                AddIDConstraint(node, true);
                AddTorqueBoxConstraint(node);
            } else if (node < nodes_ - 1) {
                AddIDConstraint(node, false);
            }

            AddFrictionConeConstraint(node);

            if (node > 0) {
                // Velocity is fixed for the initial condition, do not constrain it
                AddHolonomicConstraint(node);
                AddVelocityBoxConstraint(node);

                // The second configuration can be effected by the second velocity
                AddConfigurationBoxConstraint(node);
                AddSwingHeightConstraint(node);

                AddCollisionConstraint(node);
            }
        }

        if (constraint_triplet_idx_ != constraint_triplets_.size()) {
            std::cerr << "triplet_idx: " << constraint_triplet_idx_ << "\nconstraint triplet size: " << constraint_triplets_.size() << std::endl;
            throw std::runtime_error("Constraints did not populate the full matrix!");
        }

        A_.setFromTriplets(constraint_triplets_.begin(), constraint_triplets_.end());
    }

    void FullOrderMpc::IntegrationConstraint(const ad::ad_vector_t& dqk_dqkp1_dvk_dvkp1, const ad::ad_vector_t& dt_qkbar_qkp1bar_vk_vkp1, ad::ad_vector_t& violation) const {
        // From the reference trajectory
        const ad::adcg_t& dt = dt_qkbar_qkp1bar_vk_vkp1(0);
        const ad::ad_vector_t& qkbar = dt_qkbar_qkp1bar_vk_vkp1.segment(1, config_dim_);
        const ad::ad_vector_t& qkp1bar = dt_qkbar_qkp1bar_vk_vkp1.segment(1 + config_dim_, config_dim_);
        const ad::ad_vector_t& vkbar = dt_qkbar_qkp1bar_vk_vkp1.segment(1 + 2*config_dim_, vel_dim_);
        const ad::ad_vector_t& vkp1bar = dt_qkbar_qkp1bar_vk_vkp1.segment(1 + 2*config_dim_ + vel_dim_, vel_dim_);

        // Changes from decision variables
        const ad::ad_vector_t& dqk = dqk_dqkp1_dvk_dvkp1.head(vel_dim_);
        const ad::ad_vector_t& dqkp1 = dqk_dqkp1_dvk_dvkp1.segment(vel_dim_, vel_dim_);
        const ad::ad_vector_t& dvk = dqk_dqkp1_dvk_dvkp1.segment(2*vel_dim_, vel_dim_);
        const ad::ad_vector_t& dvkp1 = dqk_dqkp1_dvk_dvkp1.tail(vel_dim_);

        // Get the current configuration
        const ad::ad_vector_t qk = torc::models::ConvertdqToq(dqk, qkbar);
        const ad::ad_vector_t qkp1 = torc::models::ConvertdqToq(dqkp1, qkp1bar);

        // Velocity
        const ad::ad_vector_t vk = vkbar + dvk;
        const ad::ad_vector_t vkp1 = vkp1bar + dvkp1;

        const ad::ad_vector_t v = dt*0.5*(vk + vkp1);

        const ad::ad_vector_t qkp1_new = pinocchio::integrate(robot_model_->GetADPinModel(), qk, v);

        // Floating base position differences
        violation.resize(v.size());
        violation.head<POS_VARS>() = qkp1_new.head<POS_VARS>() - qkp1.head<POS_VARS>();

        // Quaternion difference in the tangent space
        Eigen::Quaternion<ad::adcg_t> quat_kp1(qkp1.segment<QUAT_VARS>(POS_VARS));
        Eigen::Quaternion<ad::adcg_t> quat_kp1_new(qkp1_new.segment<QUAT_VARS>(POS_VARS));

        // Eigen's inverse has an if statement, so we can't use it in codegen
        quat_kp1 = Eigen::Quaternion<torc::ad::adcg_t>(quat_kp1.conjugate().coeffs() / quat_kp1.squaredNorm());   // Assumes norm > 0
        violation.segment<3>(POS_VARS) = pinocchio::quaternion::log3(quat_kp1 * quat_kp1_new);

        // Joint differences
        violation.tail(v.size() - FLOATING_VEL) = qkp1_new.tail(qk.size() - FLOATING_BASE) - qkp1.tail(qk.size() - FLOATING_BASE);
    }

    void FullOrderMpc::HolonomicConstraint(const std::string& frame, const ad::ad_vector_t& dqk_dvk, const ad::ad_vector_t& qk_vk, ad::ad_vector_t& violation) const {
        const ad::ad_vector_t& dq = dqk_dvk.head(vel_dim_);
        const ad::ad_vector_t& dv = dqk_dvk.tail(vel_dim_);

        const ad::ad_vector_t& qkbar = qk_vk.head(config_dim_);
        const ad::ad_vector_t& vkbar = qk_vk.tail(vel_dim_);

        ad::ad_vector_t q = torc::models::ConvertdqToq(dq, qkbar);
        ad::ad_vector_t v = vkbar + dv;

        // forward kinematics
        pinocchio::forwardKinematics(robot_model_->GetADPinModel(), *robot_model_->GetADPinData(), q, v);

        // Get the frame velocity
        const long frame_idx = robot_model_->GetFrameIdx(frame);
        // TODO: Consider going back to LocalWorldAligned frame!
        const ad::ad_vector_t vel = pinocchio::getFrameVelocity(robot_model_->GetADPinModel(), *robot_model_->GetADPinData(), frame_idx, pinocchio::LOCAL).linear();
        // const ad::ad_vector_t vel = pinocchio::getFrameVelocity(robot_model_->GetADPinModel(), *robot_model_->GetADPinData(), frame_idx, pinocchio::LOCAL_WORLD_ALIGNED).linear();

        // TODO: In the future we will want to rotate this into the ground frame so the constraint is always tangential to the terrain

        // Violation is the velocity as we want to drive it to 0
        // TODO: Consider going back to 3
        violation = vel.head<2>();
    }

    void FullOrderMpc::SwingHeightConstraint(const std::string& frame, const ad::ad_vector_t& dqk, const ad::ad_vector_t& qk_desheight, ad::ad_vector_t& violation) const {
        const ad::ad_vector_t& qkbar = qk_desheight.head(config_dim_);
        const ad::adcg_t& des_height = qk_desheight(config_dim_);

        const ad::ad_vector_t q = torc::models::ConvertdqToq(dqk, qkbar);

        // Forward kinematics
        pinocchio::forwardKinematics(robot_model_->GetADPinModel(), *robot_model_->GetADPinData(), q);
        const long frame_idx = robot_model_->GetFrameIdx(frame);
        pinocchio::updateFramePlacement(robot_model_->GetADPinModel(), *robot_model_->GetADPinData(), frame_idx);

        // Get frame position in world frame (data oMf)
        ad::ad_vector_t frame_pos = robot_model_->GetADPinData()->oMf.at(frame_idx).translation();

        violation.resize(1);
        violation(0) = frame_pos(2) - des_height;
    }

    void FullOrderMpc::InverseDynamicsConstraint(const std::vector<std::string>& frames,
        const ad::ad_vector_t& dqk_dvk_dvkp1_dtauk_dfk, const ad::ad_vector_t& qk_vk_vkp1_tauk_fk_dt, ad::ad_vector_t& violation) const {
        // Decision variables
        const ad::ad_vector_t& dqk = dqk_dvk_dvkp1_dtauk_dfk.head(vel_dim_);
        const ad::ad_vector_t& dvk = dqk_dvk_dvkp1_dtauk_dfk.segment(vel_dim_, vel_dim_);
        const ad::ad_vector_t& dvkp1 = dqk_dvk_dvkp1_dtauk_dfk.segment(2*vel_dim_, vel_dim_);
        ad::ad_vector_t dtauk(vel_dim_);
        dtauk << ad::ad_vector_t::Zero(FLOATING_VEL), dqk_dvk_dvkp1_dtauk_dfk.segment(3*vel_dim_, input_dim_);
        const ad::ad_vector_t& dfk = dqk_dvk_dvkp1_dtauk_dfk.segment(3*vel_dim_ + input_dim_, CONTACT_3DOF*num_contact_locations_);

        // Reference trajectory
        const ad::ad_vector_t& qk = qk_vk_vkp1_tauk_fk_dt.head(config_dim_);
        const ad::ad_vector_t& vk = qk_vk_vkp1_tauk_fk_dt.segment(config_dim_, vel_dim_);
        const ad::ad_vector_t& vkp1 = qk_vk_vkp1_tauk_fk_dt.segment(config_dim_ + vel_dim_, vel_dim_);
        ad::ad_vector_t tauk(vel_dim_);
        tauk << ad::ad_vector_t::Zero(FLOATING_VEL), qk_vk_vkp1_tauk_fk_dt.segment(config_dim_ + 2*vel_dim_, input_dim_);
        const ad::ad_vector_t& fk = qk_vk_vkp1_tauk_fk_dt.segment(config_dim_ + 2*vel_dim_ + input_dim_, CONTACT_3DOF*num_contact_locations_);
        const ad::adcg_t& dt = qk_vk_vkp1_tauk_fk_dt(config_dim_ + 2*vel_dim_ + input_dim_ + CONTACT_3DOF*num_contact_locations_);

        // Current values
        const ad::ad_vector_t qk_curr = models::ConvertdqToq(dqk, qk);
        const ad::ad_vector_t vk_curr = dvk + vk;
        const ad::ad_vector_t vkp1_curr = dvkp1 + vkp1;
        const ad::ad_vector_t tauk_curr = dtauk + tauk;
        const ad::ad_vector_t fk_curr = dfk + fk;

        // Intermediate values
        const ad::ad_vector_t a = (vkp1_curr - vk_curr)/dt;
        std::vector<models::ExternalForce<ad::adcg_t>> f_ext;

        int idx = 0;
        for (const auto& frame : frames) {
            f_ext.emplace_back(frame, fk_curr.segment<CONTACT_3DOF>(idx));
            idx += CONTACT_3DOF;
        }

        // Compute error
        const ad::ad_vector_t tau_id = models::InverseDynamics(robot_model_->GetADPinModel(), *robot_model_->GetADPinData(),
            qk_curr, vk_curr, a, f_ext);

        violation = tau_id - tauk_curr;
    }

    void FullOrderMpc::SelfCollisionConstraint(const std::string& frame1, const std::string& frame2,
        const ad::ad_vector_t& dqk, const ad::ad_vector_t& qk_r1_r2, ad::ad_vector_t& violation) {
        const ad::ad_vector_t& qk = qk_r1_r2.head(config_dim_);
        const ad::ad_vector_t q = models::ConvertdqToq(dqk, qk);
        const ad::adcg_t& r1 = qk_r1_r2(config_dim_);
        const ad::adcg_t& r2 = qk_r1_r2(config_dim_ + 1);

        pinocchio::forwardKinematics(robot_model_->GetADPinModel(), *robot_model_->GetADPinData(), q);
        const long frame_idx1 = robot_model_->GetFrameIdx(frame1);
        pinocchio::updateFramePlacement(robot_model_->GetADPinModel(), *robot_model_->GetADPinData(), frame_idx1);

        const long frame_idx2 = robot_model_->GetFrameIdx(frame2);
        pinocchio::updateFramePlacement(robot_model_->GetADPinModel(), *robot_model_->GetADPinData(), frame_idx2);

        ad::ad_vector_t frame_pos1 = robot_model_->GetADPinData()->oMf.at(frame_idx1).translation();
        ad::ad_vector_t frame_pos2 = robot_model_->GetADPinData()->oMf.at(frame_idx2).translation();

        violation.resize(1);
        violation(0) = (frame_pos1 - frame_pos2).norm() - (r1 + r2);
    }

    void FullOrderMpc::FrictionConeConstraint(const ad::ad_vector_t& df, const ad::ad_vector_t& fk_margin, ad::ad_vector_t& violation) const {
        const ad::ad_vector_t f = df + fk_margin.head<CONTACT_3DOF>();
        const ad::adcg_t& margin = fk_margin(3);

        violation.resize(1);
        violation(0) = friction_coef_*f(2) - CppAD::sqrt(f(0)*f(0) + f(1)*f(1) + margin*margin);
    }


    // TODO: Consider removing this constraint
    // void FullOrderMpc::AddICConstraint() {
    //     // Set constraint matrix
    //     int row_start = 0;
    //     int col_start = 0;
    //     DiagonalScalarMatrixToTriplet(1, row_start, col_start, 2*robot_model_->GetVelDim(), constraint_triplets_, constraint_triplet_idx_);
    //
    //     // Set bounds
    //     // Configuration differences are all set to 0
    //     osqp_instance_.lower_bounds.head(robot_model_->GetVelDim()).setZero();
    //     osqp_instance_.upper_bounds.head(robot_model_->GetVelDim()).setZero();
    //
    //     // Velocity differences are all set to 0
    //     osqp_instance_.lower_bounds.segment(robot_model_->GetVelDim(), robot_model_->GetVelDim()).setZero();
    //     osqp_instance_.upper_bounds.segment(robot_model_->GetVelDim(), robot_model_->GetVelDim()).setZero();
    // }

    void FullOrderMpc::AddIntegrationConstraint(int node) {
        assert(node != nodes_ - 1);

        const int row_start = GetConstraintRow(node, Integrator);


        const vectorx_t x_zero = vectorx_t::Zero(integration_constraint_->GetDomainSize());
        vectorx_t p(integration_constraint_->GetParameterSize());
        p << dt_[node], traj_.GetConfiguration(node), traj_.GetConfiguration(node + 1), traj_.GetVelocity(node), traj_.GetVelocity(node + 1);

        matrixx_t jac;
        integration_constraint_->GetJacobian(x_zero, p, jac);

        // dqk
        if (node != 0) {
            int col_start = GetDecisionIdx(node, Configuration);
            MatrixToTripletWithSparsitySet(jac.middleCols(0, vel_dim_), row_start, col_start,
                constraint_triplets_, constraint_triplet_idx_, ws_->sp_int_dqk);

            // dvk
            col_start = GetDecisionIdx(node, Velocity);
            MatrixToTripletWithSparsitySet(jac.middleCols(2*vel_dim_, vel_dim_), row_start, col_start,
                constraint_triplets_, constraint_triplet_idx_, ws_->sp_int_dvk);
        }

        // dqkp1
        int col_start = GetDecisionIdx(node + 1, Configuration);
        MatrixToTripletWithSparsitySet(jac.middleCols(vel_dim_, vel_dim_), row_start, col_start,
            constraint_triplets_, constraint_triplet_idx_, ws_->sp_int_dqkp1);

        // dvkp1
        col_start = GetDecisionIdx(node + 1, Velocity);
        MatrixToTripletWithSparsitySet(jac.middleCols(3*vel_dim_, vel_dim_), row_start, col_start,
            constraint_triplets_, constraint_triplet_idx_, ws_->sp_int_dvkp1);

        // Lower and upper bounds
        vectorx_t y;
        integration_constraint_->GetFunctionValue(x_zero, p, y);
        osqp_instance_.lower_bounds.segment(row_start, integration_constraint_->GetRangeSize()) = -y;
        osqp_instance_.upper_bounds.segment(row_start, integration_constraint_->GetRangeSize()) = -y;
    }

    void FullOrderMpc::AddIDConstraint(int node, bool full_order) {
        assert(node != nodes_ - 1);

        const int row_start = GetConstraintRow(node, ID);

        ws_->id_config_mat.setZero();
        ws_->id_vel1_mat.setZero();
        ws_->id_vel2_mat.setZero();
        ws_->id_force_mat.setZero();

        vectorx_t y;

        // compute all derivative terms
        InverseDynamicsLinearizationAD(node, ws_->id_config_mat,
            ws_->id_vel1_mat, ws_->id_vel2_mat, ws_->id_force_mat, ws_->id_tau_mat, y);

        // Full inverse dynamics
        if (full_order) {
            // dtau_dq
            if (node != 0) {
                int col_start = GetDecisionIdx(node, Configuration);
                MatrixToTripletWithSparsitySet(ws_->id_config_mat, row_start, col_start, constraint_triplets_, constraint_triplet_idx_, ws_->sp_dq);

                // dtau_dv
                col_start = GetDecisionIdx(node, Velocity);
                MatrixToTripletWithSparsitySet(ws_->id_vel1_mat, row_start, col_start, constraint_triplets_, constraint_triplet_idx_, ws_->sp_dvk);
            }

            // dtau_dtau
            int col_start = GetDecisionIdx(node, Torque);
            MatrixToTripletWithSparsitySet(ws_->id_tau_mat, row_start, col_start, constraint_triplets_, constraint_triplet_idx_, ws_->sp_tau);

            // dtau_df
            col_start = GetDecisionIdx(node, GroundForce);
            MatrixToTripletWithSparsitySet(ws_->id_force_mat, row_start, col_start, constraint_triplets_, constraint_triplet_idx_, ws_->sp_df);

            // dtau_dv2
            col_start = GetDecisionIdx(node + 1, Velocity);
            MatrixToTripletWithSparsitySet(ws_->id_vel2_mat, row_start, col_start, constraint_triplets_, constraint_triplet_idx_, ws_->sp_dvkp1);

            osqp_instance_.lower_bounds.segment(row_start, robot_model_->GetNumInputs() + FLOATING_VEL) = -y;

            osqp_instance_.upper_bounds.segment(row_start, robot_model_->GetNumInputs() + FLOATING_VEL) = -y;
        } else {    // Dynamics only for the floating base
            // dtau_dq
            int col_start = GetDecisionIdx(node, Configuration);
            MatrixToTripletWithSparsitySet(ws_->id_config_mat.topRows<FLOATING_VEL>(), row_start, col_start,
             constraint_triplets_, constraint_triplet_idx_, ws_->sp_dq_partial);

            // dtau_dv
            col_start = GetDecisionIdx(node, Velocity);
            MatrixToTripletWithSparsitySet(ws_->id_vel1_mat.topRows<FLOATING_VEL>(), row_start, col_start,
                constraint_triplets_, constraint_triplet_idx_, ws_->sp_dvk_partial);
            // MatrixToTriplet(ws_->id_vel1_mat.topRows<FLOATING_VEL>(), row_start, col_start,
            //     constraint_triplets_, constraint_triplet_idx_);

            // dtau_dtau -- no torque enters

            // dtau_df
            col_start = GetDecisionIdx(node, GroundForce);
            MatrixToTripletWithSparsitySet(ws_->id_force_mat.topRows<FLOATING_VEL>(), row_start, col_start,
                constraint_triplets_, constraint_triplet_idx_, ws_->sp_df_partial);
            // MatrixToTriplet(ws_->id_force_mat.topRows<FLOATING_VEL>(), row_start, col_start,
            //     constraint_triplets_, constraint_triplet_idx_);

            // dtau_dv2
            col_start = GetDecisionIdx(node + 1, Velocity);
            MatrixToTripletWithSparsitySet(ws_->id_vel2_mat.topRows<FLOATING_VEL>(), row_start, col_start,
                constraint_triplets_, constraint_triplet_idx_, ws_->sp_dvkp1_partial);
            // MatrixToTriplet(ws_->id_vel2_mat.topRows<FLOATING_VEL>(), row_start, col_start,
            //     constraint_triplets_, constraint_triplet_idx_);

            // Set the bounds
            osqp_instance_.lower_bounds.segment(row_start, FLOATING_VEL) = -y.head<FLOATING_VEL>();

            osqp_instance_.upper_bounds.segment(row_start, FLOATING_VEL) = -y.head<FLOATING_VEL>();
        }
    }

    void FullOrderMpc::AddFrictionConeConstraint(int node) {
        int row_start = GetConstraintRow(node, FrictionCone);
        int col_start = GetDecisionIdx(node, GroundForce);

        // vector3_t h = {1, 0, 0};
        // vector3_t l = {0, 1, 0};
        // vector3_t n = {0, 0, 1};
        //
        // ws_->fric_cone_mat << (h - n*friction_coef_).transpose(),
        //         -(h + n*friction_coef_).transpose(),
        //         (l - n*friction_coef_).transpose(),
        //         -(l + n*friction_coef_).transpose();

        for (const auto& frame : contact_frames_) {
            // ----- Zero Force in Swing ----- //
            DiagonalScalarMatrixToTriplet((1-in_contact_[frame][node])*1, row_start, col_start, CONTACT_3DOF,
                constraint_triplets_, constraint_triplet_idx_);
            osqp_instance_.lower_bounds.segment(row_start, CONTACT_3DOF) = -(1-in_contact_[frame][node])*traj_.GetForce(node, frame);
            osqp_instance_.upper_bounds.segment(row_start, CONTACT_3DOF) = -(1-in_contact_[frame][node])*traj_.GetForce(node, frame);

            row_start += CONTACT_3DOF;

            // ----- Friction Cone ----- //
            matrixx_t jac;
            vectorx_t x_zero = vectorx_t::Zero(friction_cone_constraint_->GetDomainSize());
            vectorx_t p(friction_cone_constraint_->GetParameterSize());
            p << traj_.GetForce(node, frame), friction_margin_;
            friction_cone_constraint_->GetJacobian(x_zero, p, jac);

            const auto sparsity = friction_cone_constraint_->GetJacobianSparsityPatternSet();

            MatrixToTripletWithSparsitySet(in_contact_[frame][node]*jac, row_start, col_start, constraint_triplets_, constraint_triplet_idx_, sparsity);

            // Lower and upper bounds
            vectorx_t y;
            friction_cone_constraint_->GetFunctionValue(x_zero, p, y);
            osqp_instance_.lower_bounds(row_start) = -in_contact_[frame][node]*y(0);
            osqp_instance_.upper_bounds(row_start) = in_contact_[frame][node]*std::numeric_limits<double>::max();

            row_start += friction_cone_constraint_->GetRangeSize();
            col_start += CONTACT_3DOF;

            // TODO: I think I can remove this
            // ----- Z Force Bounds ----- //
            // DiagonalScalarMatrixToTriplet(in_contact_[frame][node]*1, row_start, col_start - 1, 1,
            //     constraint_triplets_, constraint_triplet_idx_);
            //
            // osqp_instance_.lower_bounds(row_start) = -in_contact_[frame][node]*traj_.GetForce(node, frame)(2);
            // osqp_instance_.upper_bounds(row_start) = in_contact_[frame][node]*(max_grf_ - traj_.GetForce(node, frame)(2));
            //
            // row_start += 1;

        }
    }

    void FullOrderMpc::AddConfigurationBoxConstraint(int node) {
        const int row_start = GetConstraintRow(node, ConfigBox);
        const int col_start = GetDecisionIdx(node, Configuration) + FLOATING_VEL;

        // Only apply to the joints, not the floating base

        // joint identity
        DiagonalScalarMatrixToTriplet(1, row_start, col_start, robot_model_->GetNumInputs(),
            constraint_triplets_, constraint_triplet_idx_); // TODO Should probably be num joints not inputs

        // Set configuration bounds
        osqp_instance_.lower_bounds.segment(row_start, robot_model_->GetConfigDim() - FLOATING_BASE)
            = robot_model_->GetLowerConfigLimits().tail(robot_model_->GetNumInputs())
                - traj_.GetConfiguration(node).tail(robot_model_->GetNumInputs());

        osqp_instance_.upper_bounds.segment(row_start, robot_model_->GetConfigDim() - FLOATING_BASE)
            = robot_model_->GetUpperConfigLimits().tail(robot_model_->GetNumInputs())
                - traj_.GetConfiguration(node).tail(robot_model_->GetNumInputs());
    }

    void FullOrderMpc::AddVelocityBoxConstraint(int node) {
        const int row_start = GetConstraintRow(node, VelBox);
        const int col_start = GetDecisionIdx(node, Velocity) + FLOATING_VEL;

        // Only applied to the joints, not the floating base
        DiagonalScalarMatrixToTriplet(1, row_start, col_start, robot_model_->GetVelDim() - FLOATING_VEL,
            constraint_triplets_, constraint_triplet_idx_);

        // Set velocity bounds
        osqp_instance_.lower_bounds.segment(row_start, robot_model_->GetVelDim() - FLOATING_VEL)
            = -robot_model_->GetVelocityJointLimits().tail(robot_model_->GetNumInputs())
                - traj_.GetVelocity(node).tail(robot_model_->GetNumInputs());
        osqp_instance_.upper_bounds.segment(row_start, robot_model_->GetVelDim() - FLOATING_VEL)
            = robot_model_->GetVelocityJointLimits().tail(robot_model_->GetNumInputs())
                - traj_.GetVelocity(node).tail(robot_model_->GetNumInputs());
    }

    void FullOrderMpc::AddTorqueBoxConstraint(int node) {
        const int row_start = GetConstraintRow(node, TorqueBox);
        const int col_start = GetDecisionIdx(node, Torque);

        DiagonalScalarMatrixToTriplet(1, row_start, col_start, robot_model_->GetNumInputs(), constraint_triplets_, constraint_triplet_idx_);

        // Set torque bounds
        osqp_instance_.lower_bounds.segment(row_start, robot_model_->GetNumInputs()) = -robot_model_->GetTorqueJointLimits() - traj_.GetTau(node);
        osqp_instance_.upper_bounds.segment(row_start, robot_model_->GetNumInputs()) = robot_model_->GetTorqueJointLimits() - traj_.GetTau(node);
    }

    void FullOrderMpc::AddSwingHeightConstraint(int node) {
        int row_start = GetConstraintRow(node, SwingHeight);
        const int col_start = GetDecisionIdx(node, Configuration);

        // TODO: Consider adding a little feedback controller in the constraint to bring the foot back if it off the trajectory
        //  Could also choose to track velocity instead of position.
        for (const auto& frame : contact_frames_) {
            // if (!in_contact_[frame][node]) {     // Only constrain the height if not in contact
                // Linearization
                matrixx_t jac;
                vectorx_t x_zero = vectorx_t::Zero(swing_height_constraint_[frame]->GetDomainSize());
                vectorx_t p(swing_height_constraint_[frame]->GetParameterSize());
                p << traj_.GetConfiguration(node), swing_traj_[frame][node];
                swing_height_constraint_[frame]->GetJacobian(x_zero, p, jac);

                const auto sparsity = swing_height_constraint_[frame]->GetJacobianSparsityPatternSet();

                // dqk
                MatrixToTripletWithSparsitySet(jac, row_start, col_start, constraint_triplets_, constraint_triplet_idx_, sparsity);

                // Lower and upper bounds
                vectorx_t y;
                swing_height_constraint_[frame]->GetFunctionValue(x_zero, p, y);
                osqp_instance_.lower_bounds(row_start) = -y(0);
                osqp_instance_.upper_bounds(row_start) = -y(0);

                row_start++;
            // } else {
            //     matrixx_t jac = matrixx_t::Zero(swing_height_constraint_[frame]->GetRangeSize(), swing_height_constraint_[frame]->GetDomainSize());
            //     const auto sparsity = swing_height_constraint_[frame]->GetJacobianSparsityPatternSet();
            //
            //     // dqk
            //     MatrixToTripletWithSparsitySet(jac, row_start, col_start, constraint_triplets_, constraint_triplet_idx_, sparsity);
            //
            //     // Lower and upper bounds
            //     osqp_instance_.lower_bounds(row_start) = 0;
            //     osqp_instance_.upper_bounds(row_start) = 0;
            //
            //     row_start++;
            // }

            // std::cout << "Frame " << frame << " jac: \n" << jac << std::endl;
            // std::cout << "y: " << y << std::endl;
            // robot_model_->FirstOrderFK(traj_.GetConfiguration(node));
            // vector3_t frame_pos = robot_model_->GetFrameState(frame).placement.translation();
            // std::cout << "frame z: " << frame_pos(2) << std::endl;
            // std::cout << "des swing height: " << swing_traj_[frame][node] << std::endl;
        }
    }

    void FullOrderMpc::AddHolonomicConstraint(int node) {
        int row_start = GetConstraintRow(node, Holonomic);

        for (const auto& frame : contact_frames_) {
            if (in_contact_[frame][node]) {
                matrixx_t jac;
                vectorx_t x_zero = vectorx_t::Zero(holonomic_constraint_[frame]->GetDomainSize());
                vectorx_t p(holonomic_constraint_[frame]->GetParameterSize());
                p << traj_.GetConfiguration(node), traj_.GetVelocity(node);

                holonomic_constraint_[frame]->GetJacobian(x_zero, p, jac);

                // dqk
                int col_start = GetDecisionIdx(node, Configuration);
                MatrixToTripletWithSparsitySet(jac.middleCols(0, vel_dim_), row_start, col_start,
                    constraint_triplets_, constraint_triplet_idx_, ws_->sp_hol_dqk[frame]);

                // dvk
                col_start = GetDecisionIdx(node, Velocity);
                // std::cout << "holonomic dv: \n" << jac.middleCols(vel_dim_, vel_dim_) << std::endl;
                MatrixToTripletWithSparsitySet(jac.middleCols(vel_dim_, vel_dim_), row_start, col_start,
                    constraint_triplets_, constraint_triplet_idx_, ws_->sp_hol_dvk[frame]);

                // Lower and upper bounds
                vectorx_t y;
                holonomic_constraint_[frame]->GetFunctionValue(x_zero, p, y);
                osqp_instance_.lower_bounds.segment(row_start, holonomic_constraint_[frame]->GetRangeSize()) = -y;
                osqp_instance_.upper_bounds.segment(row_start, holonomic_constraint_[frame]->GetRangeSize()) = -y;
            } else {
                matrixx_t jac = matrixx_t::Zero(holonomic_constraint_[frame]->GetRangeSize(), holonomic_constraint_[frame]->GetDomainSize());

                // dqk
                int col_start = GetDecisionIdx(node, Configuration);
                MatrixToTripletWithSparsitySet(jac.middleCols(0, vel_dim_), row_start, col_start,
                    constraint_triplets_, constraint_triplet_idx_, ws_->sp_hol_dqk[frame]);

                // dvk
                col_start = GetDecisionIdx(node, Velocity);
                MatrixToTripletWithSparsitySet(jac.middleCols(vel_dim_, vel_dim_), row_start, col_start,
                    constraint_triplets_, constraint_triplet_idx_, ws_->sp_hol_dvk[frame]);

                // Lower and upper bounds
                osqp_instance_.lower_bounds.segment(row_start, holonomic_constraint_[frame]->GetRangeSize()) = vectorx_t::Zero(holonomic_constraint_[frame]->GetRangeSize());
                osqp_instance_.upper_bounds.segment(row_start, holonomic_constraint_[frame]->GetRangeSize()) = vectorx_t::Zero(holonomic_constraint_[frame]->GetRangeSize());
            }

            row_start+=2;
        }
    }

    void FullOrderMpc::AddCollisionConstraint(int node) {
        int row_start = GetConstraintRow(node, Collision);
        const int col_start = GetDecisionIdx(node, Configuration);

        int radii_idx = 0;
        for (const auto& constraint : collision_constraints_) {
            const auto sparsity = constraint->GetJacobianSparsityPatternSet();
            matrixx_t jac;
            vectorx_t x_zero = vectorx_t::Zero(constraint->GetDomainSize());
            vectorx_t p(constraint->GetParameterSize());
            p << traj_.GetConfiguration(node), radii_[0].first, radii_[0].second;

            constraint->GetJacobian(x_zero, p, jac);
            MatrixToTripletWithSparsitySet(jac, row_start, col_start, constraint_triplets_, constraint_triplet_idx_, sparsity);

            // Bounds
            vectorx_t y;
            constraint->GetFunctionValue(x_zero, p, y);
            osqp_instance_.lower_bounds(row_start) = -y(0);
            osqp_instance_.upper_bounds(row_start) = std::numeric_limits<double>::max();

            row_start++;
            radii_idx++;
        }
    }


    // -------------------------------------- //
    // -------- Constraint Violation -------- //
    // -------------------------------------- //
    double FullOrderMpc::GetConstraintViolation(const vectorx_t &qp_res) {
        // Uses l2 norm!

        double violation = 0;
        // violation += GetICViolation(qp_res);
        for (int node = 0; node < nodes_; node++) {
            // Dynamics related constraints don't happen in the last node
            // std::cout << "Node: " << node << std::endl;
            if (node < nodes_ - 1) {
                violation += GetIntegrationViolation(qp_res, node);
                // std::cout << "Integration violation: " << GetIntegrationViolation(qp_res, node) << std::endl;
            }

            if (node < nodes_full_dynamics_) {
                violation += GetIDViolation(qp_res, node, true);
                // std::cout << "Dynamics violation: " << GetIDViolation(qp_res, node, true) << std::endl;
                violation += GetTorqueBoxViolation(qp_res, node);
            } else if (node < nodes_ - 1) {
                 violation += GetIDViolation(qp_res, node, false);
            }

            // std::cout << "Friction cone violation: " << GetFrictionViolation(qp_res, node) << std::endl;
            violation += GetFrictionViolation(qp_res, node);

            // These could conflict with the initial condition constraints
            if (node > 0) {
                // Velocity is fixed for the initial condition, do not constrain it
                // std::cout << "Holonomic violation: " << GetHolonomicViolation(qp_res, node) << std::endl;
                violation += GetHolonomicViolation(qp_res, node);
                violation += GetVelocityBoxViolation(qp_res, node);

                violation += GetConfigurationBoxViolation(qp_res, node);
                violation += GetSwingHeightViolation(qp_res, node);

                violation += GetCollisionViolation(qp_res, node);
            }
        }

        // TODO: What node discretization do I want to use?
        return dt_[1]*sqrt(violation);
    }

    // double FullOrderMpc::GetICViolation(const vectorx_t &qp_res) {
    //     return qp_res.head(robot_model_->GetVelDim()*2).squaredNorm();
    // }

    double FullOrderMpc::GetIntegrationViolation(const vectorx_t &qp_res, int node) const {

        vectorx_t dqk, dvk;
        if (node != 0) {
            dqk = qp_res.segment(GetDecisionIdx(node, Configuration), vel_dim_);
            dvk = qp_res.segment(GetDecisionIdx(node, Velocity), vel_dim_);
        } else {
            dqk = vectorx_t::Zero(vel_dim_);
            dvk = vectorx_t::Zero(vel_dim_);
        }

        const vectorx_t& dqkp1 = qp_res.segment(GetDecisionIdx(node + 1, Configuration), vel_dim_);
        const vectorx_t& dvkp1 = qp_res.segment(GetDecisionIdx(node + 1, Velocity), vel_dim_);

        vectorx_t x(integration_constraint_->GetDomainSize());
        x << dqk, dqkp1, dvk, dvkp1;

        vectorx_t p(integration_constraint_->GetParameterSize());
        p << dt_[node], traj_.GetConfiguration(node), traj_.GetConfiguration(node + 1), traj_.GetVelocity(node), traj_.GetVelocity(node + 1);

        vectorx_t y;
        integration_constraint_->GetFunctionValue(x, p, y);

        return y.squaredNorm();
    }

    double FullOrderMpc::GetIDViolation(const vectorx_t &qp_res, int node, bool full_order) {

        vectorx_t x(inverse_dynamics_constraint_->GetDomainSize());
        vectorx_t p(inverse_dynamics_constraint_->GetParameterSize());
        vectorx_t y;

        vectorx_t qp_tau = vectorx_t::Zero(input_dim_);
        if (full_order) {
            qp_tau = qp_res.segment(GetDecisionIdx(node, Torque), input_dim_);
        }

        vectorx_t dqk, dvk;
        if (node != 0) {
            dqk = qp_res.segment(GetDecisionIdx(node, Configuration), vel_dim_);
            dvk = qp_res.segment(GetDecisionIdx(node, Velocity), vel_dim_);
        } else {
            dqk = vectorx_t::Zero(vel_dim_);
            dvk = vectorx_t::Zero(vel_dim_);
        }

        x << dqk,
            dvk,
            qp_res.segment(GetDecisionIdx(node + 1, Velocity), vel_dim_),
            qp_tau,
            qp_res.segment(GetDecisionIdx(node, GroundForce), CONTACT_3DOF*num_contact_locations_);

        vectorx_t f(CONTACT_3DOF*num_contact_locations_);
        int f_idx = 0;
        for (const auto& frame : contact_frames_) {
            f.segment<CONTACT_3DOF>(f_idx) = traj_.GetForce(node, frame);
            f_idx += CONTACT_3DOF;
        }

        p << traj_.GetConfiguration(node), traj_.GetVelocity(node), traj_.GetVelocity(node + 1),
            traj_.GetTau(node), f, dt_[node];

        inverse_dynamics_constraint_->GetFunctionValue(x, p, y);

        if (full_order) {
            return y.squaredNorm();
        } else {
            return y.head<FLOATING_VEL>().squaredNorm();
        }

        // vectorx_t q;
        // ConvertdqToq(qp_res.segment(GetDecisionIdx(node, Configuration), robot_model_->GetVelDim()),
        //     traj_.GetConfiguration(node), q);
        // const vectorx_t v = qp_res.segment(GetDecisionIdx(node, Velocity), robot_model_->GetVelDim()) + traj_.GetVelocity(node);
        // const vectorx_t v2 = qp_res.segment(GetDecisionIdx(node + 1, Velocity), robot_model_->GetVelDim()) + traj_.GetVelocity(node + 1);
        // const vectorx_t a = (v2 - v)/dt_[node];
        //
        // // std::cout << "full order: " << full_order << std::endl;
        //
        // std::vector<models::ExternalForce<double>> f_ext;
        // int idx = GetDecisionIdx(node, GroundForce);
        // for (const auto& frame : contact_frames_) {
        //     f_ext.emplace_back(frame, qp_res.segment(idx, 3) + traj_.GetForce(node, frame));
        //     // std::cout << "frame: " << frame << ", force: " << (qp_res.segment(idx, 3) + traj_.GetForce(node, frame)).transpose() << std::endl;
        //     idx += 3;
        // }
        //
        // const vectorx_t tau_id = robot_model_->InverseDynamics(q, v, a, f_ext);
        //
        // // std::cout << "tau id: " << tau_id.transpose() << std::endl;
        //
        // if (full_order) {
        //     vectorx_t tau(robot_model_->GetVelDim());
        //     tau << Eigen::Vector<double, FLOATING_VEL>::Zero(),
        //         qp_res.segment(GetDecisionIdx(node, Torque), robot_model_->GetNumInputs()) + traj_.GetTau(node);
        //
        //     // std::cout << "tau dec: " << tau.transpose() << std::endl;
        //     // std::cout << std::endl;
        //
        //     return (tau - tau_id).squaredNorm();
        // } else {
        //     return tau_id.head<FLOATING_VEL>().squaredNorm();
        // }
    }

    double FullOrderMpc::GetFrictionViolation(const vectorx_t &qp_res, int node) {
        double violation = 0;
        int idx = GetDecisionIdx(node, GroundForce);
        for (const auto& frame : contact_frames_) {
            // No force constraint
            vector3_t force = qp_res.segment<3>(idx) + traj_.GetForce(node, frame);
            violation += (1-in_contact_[frame][node])*force.squaredNorm();

            if (in_contact_[frame][node]) {
                // Friction Cone constraints
                vectorx_t x(friction_cone_constraint_->GetDomainSize());
                vectorx_t p(friction_cone_constraint_->GetParameterSize());
                vectorx_t y;

                x << qp_res.segment<3>(idx);
                p << traj_.GetForce(node, frame), friction_margin_;

                friction_cone_constraint_->GetFunctionValue(x, p, y);
                // std::cout << frame << ": " << y(0) << std::endl;
                // std::cout << "force: " << force.transpose() << std::endl;
                violation += std::min(y(0), 0.)*std::min(y(0), 0.);

                // if (force(2) < 0) {
                //     violation += std::pow(force(2), 2);
                // } else if (force(2) > max_grf_) {
                //     violation += std::pow(force(2) - max_grf_, 2);
                // }
            }

            idx += 3;

        }

        // std::cout << "friction violation: " << violation << std::endl;
        return violation;
    }

    double FullOrderMpc::GetConfigurationBoxViolation(const vectorx_t &qp_res, int node) {
        vectorx_t q;
        ConvertdqToq(qp_res.segment(GetDecisionIdx(node, Configuration), robot_model_->GetVelDim()),
            traj_.GetConfiguration(node), q);

        for (int i = FLOATING_BASE; i < q.size(); i++) {
            if (q(i) < robot_model_->GetLowerConfigLimits()(i)) {
                q(i) = q(i) - robot_model_->GetLowerConfigLimits()(i);
            } else if (q(i) > robot_model_->GetUpperConfigLimits()(i)) {
                q(i) = q(i) - robot_model_->GetUpperConfigLimits()(i);
            } else {
                q(i) = 0;
            }
        }

        return q.tail(robot_model_->GetConfigDim() - FLOATING_BASE).squaredNorm();
    }

    double FullOrderMpc::GetVelocityBoxViolation(const vectorx_t &qp_res, int node) {
        vectorx_t vel = qp_res.segment(GetDecisionIdx(node, Velocity), robot_model_->GetVelDim()) + traj_.GetVelocity(node);
        for (int i = FLOATING_VEL; i < vel.size(); i++) {
            if (vel(i) < -robot_model_->GetVelocityJointLimits()(i)) {
                vel(i) = std::abs(vel(i)) - robot_model_->GetVelocityJointLimits()(i);
            } else if (vel(i) > robot_model_->GetVelocityJointLimits()(i)) {
                vel(i) = vel(i) - robot_model_->GetVelocityJointLimits()(i);
            } else {
                vel(i) = 0;
            }
        }

        return vel.tail(robot_model_->GetVelDim() - FLOATING_VEL).squaredNorm();
    }

    double FullOrderMpc::GetTorqueBoxViolation(const vectorx_t &qp_res, int node) {
        vectorx_t tau = qp_res.segment(GetDecisionIdx(node, Torque), robot_model_->GetNumInputs()) + traj_.GetTau(node);

        for (int i = 0; i < tau.size(); i++) {
            if (tau(i) < -robot_model_->GetTorqueJointLimits()(i)) {
                tau(i) = tau(i) + robot_model_->GetTorqueJointLimits()(i);
            } else if (tau(i) > robot_model_->GetTorqueJointLimits()(i)) {
                tau(i) = tau(i) - robot_model_->GetTorqueJointLimits()(i);
            } else {
                tau(i) = 0;
            }
        }

        return tau.squaredNorm();
    }

    double FullOrderMpc::GetHolonomicViolation(const vectorx_t &qp_res, int node) {
        double violation = 0;

        for (const auto& frame : contact_frames_) {
            if (in_contact_[frame][node]) {
                vectorx_t x(holonomic_constraint_[frame]->GetDomainSize());
                vectorx_t p(holonomic_constraint_[frame]->GetParameterSize());
                vectorx_t y;

                const vectorx_t& dqk = qp_res.segment(GetDecisionIdx(node, Configuration), vel_dim_);
                const vectorx_t& dvk = qp_res.segment(GetDecisionIdx(node, Velocity), vel_dim_);
                x << dqk, dvk;

                p << traj_.GetConfiguration(node), traj_.GetVelocity(node);

                holonomic_constraint_[frame]->GetFunctionValue(x, p, y);
                violation += y.squaredNorm();
            }
        }

        return violation;
    }

    double FullOrderMpc::GetCollisionViolation(const vectorx_t& qp_res, int node) {
        double violation = 0;

        int radii_idx = 0;
        for (const auto& constraint : collision_constraints_) {
            vectorx_t x(constraint->GetDomainSize());
            x << qp_res.segment(GetDecisionIdx(node, Configuration), vel_dim_);

            vectorx_t p(constraint->GetParameterSize());
            p << traj_.GetConfiguration(node), radii_[radii_idx].first, radii_[radii_idx].second;

            vectorx_t y;
            constraint->GetFunctionValue(x, p, y);

            violation += std::pow(std::min(y(0), 0.), 2);
            radii_idx++;
        }

        return violation;
    }


    double FullOrderMpc::GetSwingHeightViolation(const vectorx_t &qp_res, int node) {
        double violation = 0;
        for (const auto& frame : contact_frames_) {
            vectorx_t x(swing_height_constraint_[frame]->GetDomainSize());
            vectorx_t p(swing_height_constraint_[frame]->GetParameterSize());
            vectorx_t y;

            const vectorx_t& dqk = qp_res.segment(GetDecisionIdx(node, Configuration), vel_dim_);
            x << dqk;

            p << traj_.GetConfiguration(node), swing_traj_[frame][node];

            swing_height_constraint_[frame]->GetFunctionValue(x, p, y);
            violation += y.squaredNorm();
        }

        return violation;
    }


    // -------------------------------------- //
    // -------- Linearization Helpers ------- //
    // -------------------------------------- //
    // matrix3_t FullOrderMpc::QuatIntegrationLinearizationXi(int node) {
    //     assert(node != nodes_ - 1);
    //     constexpr double DELTA = 1e-8;
    //
    //     // TODO: Change for code gen derivative
    //     quat_t qbar_kp1 = traj_.GetQuat(node+1);
    //     quat_t qbar_k = traj_.GetQuat(node);
    //     vector3_t xi = vector3_t::Zero();
    //
    //     // Use the velocity from the current and next node
    //     vector3_t wk = traj_.GetVelocity(node).segment<3>(POS_VARS);
    //     vector3_t wkp1 = traj_.GetVelocity(node + 1).segment<3>(POS_VARS);
    //     vector3_t w = 0.5*(wk + wkp1);
    //
    //     const vector3_t xi1_kp1 = robot_model_->QuaternionIntegrationRelative(qbar_kp1, qbar_k, xi, w, dt_[node]);
    //     matrix3_t update_fd = matrix3_t::Zero();
    //     for (int i = 0; i < 3; i++) {
    //         xi(i) += DELTA;
    //         vector3_t xi2_kp1 = robot_model_->QuaternionIntegrationRelative(qbar_kp1, qbar_k, xi, w, dt_[node]);
    //         update_fd.col(i) = (xi2_kp1 - xi1_kp1)/DELTA;
    //         xi(i) -= DELTA;
    //     }
    //
    //     return update_fd;
    // }

    // matrix3_t FullOrderMpc::QuatIntegrationLinearizationW(int node) {
    //     assert(node != nodes_ - 1);
    //
    //     // TODO: Change for code gen derivative
    //     quat_t qbar_kp1 = traj_.GetQuat(node+1);
    //     quat_t qbar_k = traj_.GetQuat(node);
    //     vector3_t xi = vector3_t::Zero();
    //     // TODO: Verify that this is in the correct frame!
    //     // Use the velocity from the current and next node
    //     vector3_t wk = traj_.GetVelocity(node).segment<3>(POS_VARS);
    //     vector3_t wkp1 = traj_.GetVelocity(node + 1).segment<3>(POS_VARS);
    //     vector3_t w = 0.5*(wk + wkp1);
    //
    //     const vector3_t xi1_kp1 = robot_model_->QuaternionIntegrationRelative(qbar_kp1, qbar_k, xi, w, dt_[node]);
    //     matrix3_t update_fd = matrix3_t::Zero();
    //     for (int i = 0; i < 3; i++) {
    //         w(i) += FD_DELTA;
    //         vector3_t xi2_kp1 = robot_model_->QuaternionIntegrationRelative(qbar_kp1, qbar_k, xi, w, dt_[node]);
    //         for (int j = 0; j < 3; j++) {
    //             update_fd(j, i) = (xi2_kp1(j) - xi1_kp1(j))/FD_DELTA;
    //         }
    //
    //         w(i) -= FD_DELTA;
    //     }
    //
    //     return 0.5*update_fd;
    // }

    void FullOrderMpc::InverseDynamicsLinearizationAD(int node, matrixx_t& dtau_dq, matrixx_t& dtau_dv1, matrixx_t& dtau_dv2, matrixx_t& dtau_df, matrixx_t& dtau, vectorx_t& y) {
        // Linearizing about the nominal trajectory
        const vectorx_t x = vectorx_t::Zero(inverse_dynamics_constraint_->GetDomainSize());

        vectorx_t f(CONTACT_3DOF * num_contact_locations_);
        int f_idx = 0;
        for (const auto& frame : contact_frames_) {
            f.segment<CONTACT_3DOF>(f_idx) = traj_.GetForce(node, frame);
            f_idx += CONTACT_3DOF;
        }

        vectorx_t p(inverse_dynamics_constraint_->GetParameterSize());
        p << traj_.GetConfiguration(node), traj_.GetVelocity(node), traj_.GetVelocity(node + 1), traj_.GetTau(node), f, dt_[node];

        matrixx_t jac;
        inverse_dynamics_constraint_->GetJacobian(x, p, jac);

        dtau_dq = jac.middleCols(0, vel_dim_);
        dtau_dv1 = jac.middleCols(vel_dim_, vel_dim_);
        dtau_dv2 = jac.middleCols(2*vel_dim_, vel_dim_);
        dtau = jac.middleCols(3*vel_dim_, input_dim_);
        dtau_df =  jac.middleCols(3*vel_dim_ + input_dim_, CONTACT_3DOF*num_contact_locations_);

        inverse_dynamics_constraint_->GetFunctionValue(x, p, y);
    }


    void FullOrderMpc::InverseDynamicsLinearizationAnalytic(int node, matrixx_t& dtau_dq, matrixx_t& dtau_dv1,
        matrixx_t& dtau_dv2, matrixx_t& dtau_df) {
        assert(node != nodes_ - 1);

        // Compute acceleration via finite difference
        ws_->acc = (traj_.GetVelocity(node + 1) - traj_.GetVelocity(node))/dt_[node];
        // std::cout << "mpc a: " << a.transpose() << std::endl;
        // std::cout << "mpc dt: " << dt_[node] << std::endl;

        // TODO: Use the ws_->f_ext
        // Get external forces
        std::vector<models::ExternalForce<double>> f_ext;
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


    // matrix43_t FullOrderMpc::QuatLinearization(int node) {
    //     matrix43_t q_lin;
    //
    //     quat_t qbar = traj_.GetQuat(node);
    //     quat_t q_pert;
    //     vector3_t pert = vector3_t::Zero();
    //     for (int i = 0; i < 3; i++) {
    //         pert(i) += FD_DELTA;
    //         pinocchio::quaternion::exp3(pert, q_pert);
    //         pert(i) -= FD_DELTA;
    //         q_pert = qbar*q_pert;
    //
    //         q_lin(0, i) = (q_pert.x() - qbar.x())/FD_DELTA;
    //         q_lin(1, i) = (q_pert.y() - qbar.y())/FD_DELTA;
    //         q_lin(2, i) = (q_pert.z() - qbar.z())/FD_DELTA;
    //         q_lin(3, i) = (q_pert.w() - qbar.w())/FD_DELTA;
    //     }
    //
    //     return q_lin;
    // }

    // void FullOrderMpc::SwingHeightLinearization(int node, const std::string& frame, matrix6x_t& jacobian) {
    //     // The world frame seems to behave better numerically, but I really want the LOCAL_WORLD_ALIGNED, not WORLD.
    //     // TODO: Speed up by not calling ComputeJointJacobian each time.
    //     robot_model_->GetFrameJacobian(frame, traj_.GetConfiguration(node), jacobian, pinocchio::LOCAL_WORLD_ALIGNED);
    //     // std::cout << "frame: " << frame << std::endl;
    //     // std::cout << "local world aligned: \n" << jacobian << std::endl;
    //
    //     // *** Note *** The pinocchio body velocity is in the local frame, put I want perturbations to
    //     // configurations in the world frame, so we can always set the first 3x3 mat to identity
    //     jacobian.topLeftCorner<3,3>().setIdentity();
    // }

    // void FullOrderMpc::HolonomicLinearizationq(int node, const std::string& frame, matrix6x_t& jacobian) {
    //     robot_model_->FrameVelDerivWrtConfiguration(traj_.GetConfiguration(node),
    //         traj_.GetVelocity(node), vectorx_t::Zero(robot_model_->GetVelDim()), frame, jacobian);
    // }

    // void FullOrderMpc::HolonomicLinearizationv(int node, const std::string& frame, matrix6x_t& jacobian) {
    //     robot_model_->GetFrameJacobian(frame, traj_.GetConfiguration(node), jacobian, pinocchio::LOCAL_WORLD_ALIGNED);
    // }


    // ------------------------------------------------- //
    // ----------------- Cost Creation ----------------- //
    // ------------------------------------------------- //
    void FullOrderMpc::CreateCostPattern() {
        for (int i = 0; i < nodes_; i++) {
            for (auto& data :  cost_data_) {
                if (data.type == CostTypes::Configuration && i != 0) {
                    int decision_idx = GetDecisionIdx(i, Configuration);
                    data.sp_pattern = cost_.GetGaussNewtonSparsityPattern(data.constraint_name);
                    AddSparsitySet(data.sp_pattern, decision_idx, decision_idx, objective_triplets_);
                } else if (data.type == CostTypes::VelocityTracking && i != 0) {
                    int decision_idx = GetDecisionIdx(i, Velocity);
                    data.sp_pattern = cost_.GetHessianSparsityPattern(data.constraint_name);
                    AddSparsitySet(data.sp_pattern, decision_idx, decision_idx, objective_triplets_);
                } else if (data.type == CostTypes::ForceReg) {
                    int decision_idx = GetDecisionIdx(i, GroundForce);
                    for (const auto& frame: contact_frames_) {
                        data.sp_pattern = cost_.GetHessianSparsityPattern(data.constraint_name);
                        AddSparsitySet(data.sp_pattern, decision_idx, decision_idx, objective_triplets_);
                        decision_idx += CONTACT_3DOF;
                    }
                } else if (data.type == CostTypes::TorqueReg && i < nodes_full_dynamics_) {
                    int decision_idx = GetDecisionIdx(i, Torque);
                    data.sp_pattern = cost_.GetHessianSparsityPattern(data.constraint_name);
                    AddSparsitySet(data.sp_pattern, decision_idx, decision_idx, objective_triplets_);
                } else if (data.type == CostTypes::ForwardKinematics && i > 0) {
                    int decision_idx = GetDecisionIdx(i, Configuration);
                    data.sp_pattern = cost_.GetGaussNewtonSparsityPattern(data.constraint_name);
                    AddSparsitySet(data.sp_pattern, decision_idx, decision_idx, objective_triplets_);
                }
            }
        }

        osqp_instance_.objective_matrix.setFromTriplets(objective_triplets_.begin(), objective_triplets_.end());

        // ----- PSD Checks
        // std::cout << "sparsity pattern mat: \n" << osqp_instance_.objective_matrix << std::endl;
        //
        // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(osqp_instance_.objective_matrix.toDense());
        // if (solver.info() == Eigen::Success) {
        //     std::cout << "Eigenvalues are:\n" << solver.eigenvalues() << std::endl;
        // } else {
        //     std::cout << "Eigenvalue computation failed!" << std::endl;
        // }
        //
        // if (!osqp_instance_.objective_matrix.isApprox(osqp_instance_.objective_matrix.transpose())) {
        //     throw std::runtime_error("[Cost Function] osqp_instance_.objective_matrix is not symmetric!");
        // }
        // const auto ldlt = osqp_instance_.objective_matrix.toDense().template selfadjointView<Eigen::Upper>().ldlt();
        // if (ldlt.info() == Eigen::NumericalIssue || !ldlt.isPositive()) {
        //     throw std::runtime_error("[Cost Function] osqp_instance_.objective_matrix is not PSD!");
        // }

        ws_->obj_config_vector.resize(robot_model_->GetVelDim());
        ws_->obj_vel_vector.resize(robot_model_->GetVelDim());
        ws_->obj_tau_vector.resize(robot_model_->GetNumInputs());
    }

    void FullOrderMpc::UpdateCost() {
        objective_triplet_idx_ = 0;
        osqp_instance_.objective_vector.setZero();

        // For now the target torque will always be 0
        for (int node = 0; node < nodes_; node++) {
            double scaling = (scale_cost_ ? dt_[node] : 1);
            if (node == nodes_ - 1) {
                scaling *= terminal_cost_weight_;
            }

            for (const auto& data : cost_data_) {
                if (data.type == CostTypes::Configuration && node != 0) {
                    int decision_idx = GetDecisionIdx(node, Configuration);

                    cost_.GetApproximation(traj_.GetConfiguration(node), GetConfigTarget(node),
                                           ws_->obj_config_vector, ws_->obj_config_mat, data.constraint_name);

                    osqp_instance_.objective_vector.segment(decision_idx, robot_model_->GetVelDim()) =
                            scaling * ws_->obj_config_vector;

                    MatrixToTripletWithSparsitySet(scaling * ws_->obj_config_mat, decision_idx, decision_idx, objective_triplets_,
                                                   objective_triplet_idx_, data.sp_pattern);
                } else if (data.type == CostTypes::VelocityTracking  && node != 0) {
                    int decision_idx = GetDecisionIdx(node, Velocity);

                    cost_.GetApproximation(traj_.GetVelocity(node), GetVelTarget(node),
                                           ws_->obj_vel_vector, ws_->obj_vel_mat, data.constraint_name);

                    osqp_instance_.objective_vector.segment(decision_idx, robot_model_->GetVelDim()) =
                            scaling * ws_->obj_vel_vector;

                    MatrixToTripletWithSparsitySet(scaling * ws_->obj_vel_mat, decision_idx, decision_idx, objective_triplets_,
                                                   objective_triplet_idx_, data.sp_pattern);
                } else if (data.type == CostTypes::ForceReg) {
                    int force_idx = GetDecisionIdx(node, GroundForce);
                    for (const auto& frame: contact_frames_) {
                        vector3_t force_target = GetForceTarget(node, frame);

                        cost_.GetApproximation(traj_.GetForce(node, frame), force_target,
                                               ws_->obj_force_vector, ws_->obj_force_mat, data.constraint_name);

                        osqp_instance_.objective_vector.segment(force_idx, CONTACT_3DOF) =
                                scaling * ws_->obj_force_vector;

                        MatrixToTripletWithSparsitySet(scaling * ws_->obj_force_mat, force_idx, force_idx, objective_triplets_,
                                                       objective_triplet_idx_, data.sp_pattern);

                        force_idx += CONTACT_3DOF;
                    }
                } else if (data.type == CostTypes::TorqueReg && node < nodes_full_dynamics_) {
                    int decision_idx = GetDecisionIdx(node, Torque);
                    vectorx_t tau_target = GetTorqueTarget(node);

                    cost_.GetApproximation(traj_.GetTau(node), tau_target,
                                           ws_->obj_tau_vector, ws_->obj_tau_mat, data.constraint_name);

                    osqp_instance_.objective_vector.segment(decision_idx, robot_model_->GetNumInputs()) =
                            scaling * ws_->obj_tau_vector;

                    MatrixToTripletWithSparsitySet(scaling * ws_->obj_tau_mat, decision_idx, decision_idx, objective_triplets_,
                                                   objective_triplet_idx_, data.sp_pattern);
                } else if (data.type == CostTypes::ForwardKinematics && node > 0) {

                    // TODO: Only activate for the first stance + swing (and possibly even second stance)

                    int decision_idx = GetDecisionIdx(node, Configuration);

                    vectorx_t linear_term;
                    matrixx_t hess_term;

                    cost_.GetApproximation(traj_.GetConfiguration(node), GetDesiredFramePos(node, data.frame_name),
                                           linear_term, hess_term, data.constraint_name);

                    // std::cout << "----- " << data.constraint_name << std::endl;
                    // std::cout << "q: " << traj_.GetConfiguration(node).transpose() << std::endl;
                    // std::cout << "Hess: \n" << hess_term << std::endl;
                    // std::cout << "Linear: \n" << linear_term.transpose() << std::endl;

                    osqp_instance_.objective_vector.segment(decision_idx, robot_model_->GetVelDim()) =
                            scaling * linear_term;

                    MatrixToTripletWithSparsitySet(scaling * hess_term, decision_idx, decision_idx, objective_triplets_,
                                                   objective_triplet_idx_, data.sp_pattern);
                }
            }
        }

        objective_mat_.setFromTriplets(objective_triplets_.begin(), objective_triplets_.end());

        // std::cout << "obj mat:\n" << objective_mat_ << std::endl;
        // std::cout << "obj vec:\n" << osqp_instance_.objective_vector << std::endl;
        if (objective_triplet_idx_ != objective_triplets_.size()) {
             std::cerr << "triplet idx: " << objective_triplet_idx_ << std::endl;
             std::cerr << "triplet size: " << objective_triplets_.size() << std::endl;
            throw std::runtime_error("[Cost Function] Could not populate the cost function matrix correctly.");
        }

        // ----- PSD Checks
        // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(objective_mat_.toDense());
        // if (solver.info() == Eigen::Success) {
        //     std::cout << "Eigenvalues are:\n" << solver.eigenvalues() << std::endl;
        // } else {
        //     std::cout << "Eigenvalue computation failed!" << std::endl;
        // }
        //
        // if (!objective_mat_.isApprox(objective_mat_.transpose())) {
        //     throw std::runtime_error("[Cost Function] objective_mat_ is not symmetric!");
        // }
        // const auto ldlt = objective_mat_.toDense().template selfadjointView<Eigen::Upper>().ldlt();
        // if (ldlt.info() == Eigen::NumericalIssue || !ldlt.isPositive()) {
        //     throw std::runtime_error("[Cost Function] objective_mat_ is not PSD!");
        // }
    }

    double FullOrderMpc::GetFullCost(const vectorx_t& qp_res) {
        double cost = 0;


        for (int node = 0; node < nodes_; node++) {
            double scale = (scale_cost_ ? dt_[node] : 1);

            double cost_node = 0;

            for (const auto& data : cost_data_) {
                if (data.type == CostTypes::Configuration && node != 0) {
                    cost_node += scale * cost_.GetTermCost(
                            qp_res.segment(GetDecisionIdx(node, Configuration), robot_model_->GetVelDim()),
                            traj_.GetConfiguration(node), GetConfigTarget(node), data.constraint_name);
                } else if (data.type == CostTypes::VelocityTracking && node != 0) {
                    cost_node += scale * cost_.GetTermCost(
                            qp_res.segment(GetDecisionIdx(node, Velocity), robot_model_->GetVelDim()),
                            traj_.GetVelocity(node), GetVelTarget(node), data.constraint_name);
                } else if (data.type == CostTypes::TorqueReg && node < nodes_full_dynamics_) {
                    vectorx_t torque_target = GetTorqueTarget(node);
                    cost_node += scale * cost_.GetTermCost(qp_res.segment(GetDecisionIdx(node, Torque), robot_model_->GetNumInputs()),
                                                                                    traj_.GetTau(node), torque_target, data.constraint_name);
                } else if (data.type == CostTypes::ForceReg) {
                    int force_idx = GetDecisionIdx(node, GroundForce);
                    for (const auto& frame: contact_frames_) {
                        vector3_t force_target = GetForceTarget(node, frame);
                        cost_node += scale * cost_.GetTermCost(qp_res.segment(
                                force_idx, CONTACT_3DOF), traj_.GetForce(node, frame), force_target, data.constraint_name);
                        force_idx += CONTACT_3DOF;
                    }
                } else if (data.type == CostTypes::ForwardKinematics && node > 0) {
                    vectorx_t pos_target = GetDesiredFramePos(node, data.frame_name);
                    cost_node += scale * cost_.GetTermCost(qp_res.segment(GetDecisionIdx(node, Configuration), robot_model_->GetVelDim()),
                                                           traj_.GetConfiguration(node), pos_target, data.constraint_name);
                }
            }

            // Weight the terminal cost by a different amount
            if (node == nodes_ - 1) {
                cost_node *= terminal_cost_weight_;
            }

            cost += cost_node;
        }

        return cost;
    }

    double FullOrderMpc::GetTrajCost(const torc::mpc::Trajectory& traj, const CostTargets& targets) const {
        // TODO: If dt_ changes, then we will need to make this thread safe
        double cost = 0;

        const vectorx_t q_zero = vectorx_t::Zero(robot_model_->GetVelDim());
        const vectorx_t v_zero = vectorx_t::Zero(robot_model_->GetVelDim());
        const vectorx_t tau_zero = vectorx_t::Zero(robot_model_->GetNumInputs());
        const vector3_t force_zero = vector3_t::Zero();

        for (int node = 0; node < traj.GetNumNodes(); node++) {
            double scale = (scale_cost_ ? dt_[node] : 1);

            double cost_node = 0;

            for (const auto& data : targets.cost_data) {
                if (data.type == CostTypes::Configuration) {
                    cost_node += scale * cost_.GetTermCost(
                            q_zero,
                            traj.GetConfiguration(node), targets.q_targets.at(data.constraint_name)[node], data.constraint_name);
                } else if (data.type == CostTypes::VelocityTracking) {
                    cost_node += scale * cost_.GetTermCost(
                            v_zero,
                            traj.GetVelocity(node), targets.v_targets.at(data.constraint_name)[node], data.constraint_name);
                } else if (data.type == CostTypes::TorqueReg && node < nodes_full_dynamics_) {
                    cost_node += scale * cost_.GetTermCost(
                            tau_zero,
                            traj.GetTau(node), targets.tau_targets.at(data.constraint_name)[node], data.constraint_name);
//                } else if (data.type == CostTypes::ForceReg) {
//                    for (const auto& frame: contact_frames_) {
                        // TODO: Put back when we have forces from the mujoco sim
//                        cost_node += scale * cost_.GetTermCost(force_zero, traj.GetForce(node, frame),
//                                                               targets.force_targets.at(data.constraint_name).at(frame)[node],
//                                                               data.constraint_name);
//                    }
                } else if (data.type == CostTypes::ForwardKinematics) {
                    cost_node += scale * cost_.GetTermCost(
                            q_zero,
                            traj.GetConfiguration(node), targets.fk_targets.at(data.constraint_name)[node], data.constraint_name);
                }
            }

            // Weight the terminal cost by a different amount
            if (node == nodes_ - 1) {
                cost_node *= terminal_cost_weight_;
            }

            cost += cost_node;
        }

        return cost;
    }

    CostTargets FullOrderMpc::GetCostSnapShot() {
        CostTargets targets;

        for (const auto& data : cost_data_) {
            if (data.type == CostTypes::Configuration) {
                targets.q_targets.insert(std::pair<std::string, std::vector<vectorx_t>>(data.constraint_name, std::vector<vectorx_t>()));
            } else if (data.type == CostTypes::VelocityTracking) {
                targets.v_targets.insert(std::pair<std::string, std::vector<vectorx_t>>(data.constraint_name, std::vector<vectorx_t>()));
            } else if (data.type == CostTypes::TorqueReg) {
                targets.tau_targets.insert(std::pair<std::string, std::vector<vectorx_t>>(data.constraint_name, std::vector<vectorx_t>()));
            } else if (data.type == CostTypes::ForceReg) {
                targets.force_targets.insert(std::pair<std::string, std::map<std::string, std::vector<vector3_t>>>(
                        data.constraint_name, std::map<std::string, std::vector<vector3_t>>()));
                for (const auto& frame : contact_frames_) {
                    targets.force_targets.at(data.constraint_name).insert(std::pair<std::string, std::vector<vector3_t>>(frame, std::vector<vector3_t>()));
                }
            } else if (data.type == CostTypes::ForwardKinematics) {
                targets.fk_targets.insert(std::pair<std::string, std::vector<vector3_t>>(data.constraint_name, std::vector<vector3_t>()));
            } else {
                throw std::runtime_error("Invalid cost type!");
            }
        }

        for (int node = 0; node < nodes_; node++) {
            for (const auto& data : cost_data_) {
                if (data.type == CostTypes::Configuration) {
                    targets.q_targets[data.constraint_name].push_back(GetConfigTarget(node));
                } else if (data.type == CostTypes::VelocityTracking) {
                    targets.v_targets[data.constraint_name].push_back(GetVelTarget(node));
                } else if (data.type == CostTypes::TorqueReg) {
                    targets.tau_targets[data.constraint_name].push_back(GetTorqueTarget(node));
                } else if (data.type == CostTypes::ForceReg) {
                    for (const auto& frame : contact_frames_) {
                        targets.force_targets[data.constraint_name][frame].push_back(GetForceTarget(node, frame));
                    }
                } else if (data.type == CostTypes::ForwardKinematics) {
                    targets.fk_targets[data.constraint_name].push_back(GetDesiredFramePos(node, data.frame_name));
                }
            }
        }

        targets.cost_data = cost_data_;

        return targets;
    }

    vectorx_t FullOrderMpc::GetTorqueTarget(int node) {
        std::lock_guard<std::mutex> lock(target_mut_);
        // TODO: Put back - might want to make the target a value interpolated further into the traj
        // TODO: Might want to make this one have different weights
        // if (node == 0) {
        //     return traj_.GetTau(1);
        // }

        return vectorx_t::Zero(robot_model_->GetNumInputs());
    }

    vector3_t FullOrderMpc::GetForceTarget(int node, const std::string& frame) {
        std::lock_guard<std::mutex> lock(target_mut_);
        int num_contacts = GetNumContacts(node);
        vector3_t force_target;
        if (num_contacts != 0) {
            force_target << 0, 0, in_contact_.at(frame)[node] * 9.81 * robot_model_->GetMass() / num_contacts;
        } else {
            force_target = vector3_t::Zero();
        }

        return force_target;
    }

    vector3_t FullOrderMpc::GetDesiredFramePos(int node, std::string frame) {
        std::lock_guard<std::mutex> lock(target_mut_);
        vector3_t frame_pos;

        // TODO: Implement raibert heuristic
        frame_pos.setZero();

        return frame_pos;
    }

    // TODO: Deal with thread safety of these targets better
    vectorx_t FullOrderMpc::GetConfigTarget(int node) {
        std::lock_guard<std::mutex> lock(target_mut_);
        if (q_target_.has_value()) {
            return q_target_.value()[node];
        } else {
            throw std::runtime_error("No configuration target provided!");
        }
    }

    vectorx_t FullOrderMpc::GetVelTarget(int node) {
        std::lock_guard<std::mutex> lock(target_mut_);
        if (v_target_.has_value()) {
            return v_target_.value()[node];
        } else {
            throw std::runtime_error("No velocity target provided!");
        }
    }

    void FullOrderMpc::ParseCostYaml(const YAML::Node& cost_settings) {
        if (!cost_settings.IsSequence()) {
            throw std::runtime_error("costs must be a sequence!");
        }

        cost_data_.resize(cost_settings.size());
        int idx = 0;
        for (const auto& cost_term : cost_settings) {
            if (cost_term["type"]) {
                const std::string type = cost_term["type"].as<std::string>();
                if (type == "ConfigurationTracking") {
                    cost_data_[idx].type = CostTypes::Configuration;
                } else if (type == "VelocityTracking") {
                    cost_data_[idx].type = CostTypes::VelocityTracking;
                } else if (type == "TorqueRegularization") {
                    cost_data_[idx].type = CostTypes::TorqueReg;
                } else if (type == "ForceRegularization") {
                    cost_data_[idx].type = CostTypes::ForceReg;
                } else if (type == "ForwardKinematics") {
                    cost_data_[idx].type = CostTypes::ForwardKinematics;
                } else {
                    throw std::runtime_error("Provided cost type does not exist!");
                }
            } else {
                throw std::runtime_error("Cost term must include a type!");
            }

            if (cost_term["name"]) {
                cost_data_[idx].constraint_name = cost_term["name"].as<std::string>();
            } else {
                throw std::runtime_error("Cost term must include a name!");
            }

            if (cost_term["weight"]) {
                std::vector<double> weight = cost_term["weight"].as<std::vector<double>>();
                cost_data_[idx].weight = utils::StdToEigenVector(weight);
            } else {
                throw std::runtime_error("Cost term must include a weight!");
            }

            if (cost_term["frame"] && cost_data_[idx].type == ForwardKinematics) {
                cost_data_[idx].frame_name = cost_term["frame"].as<std::string>();
            }

            idx++;
        }

        if (cost_data_.size() == 0) {
            throw std::runtime_error("No cost terms chosen!");
        }
    }

    void FullOrderMpc::ParseCollisionYaml(const YAML::Node& collision_settings) {
        if (!collision_settings.IsSequence()) {
            throw std::runtime_error("Collision settings must be a sequence!");
        }

        for (const auto& collision : collision_settings) {
            const auto frame1 = collision["frame1"].as<std::string>();
            const auto frame2 = collision["frame2"].as<std::string>();

            const auto r1 = collision["radius1"].as<double>();
            const auto r2 = collision["radius2"].as<double>();

            collision_constraints_.emplace_back(std::make_unique<ad::CppADInterface>(
            std::bind(&FullOrderMpc::SelfCollisionConstraint, this, frame1, frame2, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
            name_ + "_" + frame1 + "_" + frame2 + "_inverse_dynamics_constraint",
            deriv_lib_path_,
            ad::DerivativeOrder::FirstOrder, vel_dim_,
            config_dim_ + 2,
            compile_derivatves_
            ));

            radii_.emplace_back(r1, r2);
        }

    }


    std::pair<double, double> FullOrderMpc::LineSearch(const vectorx_t& qp_res) {
    // TODO: Fix with the new constraint violation functions
        // Backtracing linesearch (see ETH paper)
        // TODO: Speed up

        alpha_ = 1;

        // std::cout << "------ alpha: " << 0 << " ------" << std::endl;
        // TODO: Make sure there are no bugs because I feel like I reject a lot of steps

        // Print out QP violation
//        vectorx_t constraint_vals = A_*qp_res;
//
//        double qp_constraint_violation = 0;
//        for (int i = 0; i < constraint_vals.size(); i++) {
//            if (constraint_vals(i) < osqp_instance_.lower_bounds(i)) {
//                qp_constraint_violation += std::pow(constraint_vals(i) - osqp_instance_.lower_bounds(i), 2);
//            } else if (constraint_vals(i) > osqp_instance_.upper_bounds(i)) {
//                qp_constraint_violation += std::pow(constraint_vals(i) - osqp_instance_.upper_bounds(i), 2);
//            }
//        }
//
//        std::cout << "QP constraint violation (no scaling): " << std::sqrt(qp_constraint_violation) << std::endl;

        double theta_k = GetConstraintViolation(vectorx_t::Zero(qp_res.size()));
        double phi_k = GetFullCost(vectorx_t::Zero(qp_res.size()));

//        std::cout << "Starting violation: " << theta_k << std::endl;

        while (alpha_ > ls_alpha_min_) {
            double theta_kp1 = GetConstraintViolation(alpha_*qp_res);

//            std::cout << "------ alpha: " << alpha_ << " ------" << std::endl;
//            std::cout << "violation: " << theta_kp1 << std::endl;


            if (theta_kp1 >= ls_theta_max_) {
                if (theta_kp1 < (1 - ls_gamma_theta_)*theta_k) {
                    ls_condition_ = ConstraintViolation;
                    double phi_kp1 = GetFullCost(alpha_*qp_res);
                    return std::make_pair(theta_kp1, phi_kp1);
                }
            } else if (std::max(theta_k, theta_kp1) < ls_theta_min_ && osqp_instance_.objective_vector.dot(qp_res) < 0) {
                double phi_kp1 = GetFullCost(alpha_*qp_res);
                if (phi_kp1 < (phi_k + ls_eta_*alpha_*osqp_instance_.objective_vector.dot(qp_res))) {
                    ls_condition_ = CostReduction;
                    return std::make_pair(theta_kp1, phi_kp1);
                }
            } else {
                double phi_kp1 = GetFullCost(alpha_*qp_res);
                if (phi_kp1 < (1 - ls_gamma_phi_)*phi_k || theta_kp1 < (1 - ls_gamma_theta_)*theta_k) {
                    ls_condition_ = Both;
                    return std::make_pair(theta_kp1, phi_kp1);
                }
            }
            alpha_ = ls_gamma_alpha_*alpha_;
        }
        ls_condition_ = MinAlpha;
        alpha_ = 0;

        return std::make_pair(theta_k, phi_k);
    }

    int FullOrderMpc::GetNumContacts(int node) const {
        int num_contacts = 0;
        for (const auto& frame : contact_frames_) {
            num_contacts += in_contact_.at(frame)[node];
        }

        return num_contacts;
    }


    // ------------------------------------------------- //
    // ----------- Sparsity Pattern Creation ----------- //
    // ------------------------------------------------- //
    void FullOrderMpc::CreateConstraintSparsityPattern() {
        // Fill out the triplets with dummy values -- this will allocate all the memory for the triplets
        // AddICPattern();

        for (int node = 0; node < nodes_; node++) {
            // Dynamics related constraints don't happen in the last node
            if (node < nodes_ - 1) {
                AddIntegrationPattern(node);
            }
            if (node < nodes_full_dynamics_) {
                AddIDPattern(node, true);
                AddTorqueBoxPattern(node);
            } else if (node < nodes_ - 1) {
                AddIDPattern(node, false);
            }

            AddFrictionConePattern(node);

            if (node > 0) {
                // Velocity is fixed for the initial condition, do not constrain it
                AddHolonomicPattern(node);
                AddVelocityBoxPattern(node);

                // The second configuration can be effected by the second velocity
                AddConfigurationBoxPattern(node);
                AddSwingHeightPattern(node);

                AddCollisionPattern(node);
            }
        }

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

    // void FullOrderMpc::AddICPattern() {
    //     int row_start = 0;
    //     int col_start = 0;
    //     matrixx_t id;
    //     id.setIdentity(robot_model_->GetVelDim(), robot_model_->GetVelDim());
    //     MatrixToNewTriplet(id, row_start, col_start, constraint_triplets_);
    //
    //     row_start += robot_model_->GetVelDim();
    //     col_start += robot_model_->GetVelDim();
    //
    //     MatrixToNewTriplet(id, row_start, col_start, constraint_triplets_);
    // }

    void FullOrderMpc::AddIntegrationPattern(int node) {
        assert(node != nodes_ - 1);

        const int row_start = GetConstraintRow(node, Integrator);

        const auto full_sparsity = integration_constraint_->GetJacobianSparsityPatternSet();
        ws_->sp_int_dqk = ad::CppADInterface::GetSparsityPatternCols(full_sparsity, 0, vel_dim_);
        ws_->sp_int_dqkp1 = ad::CppADInterface::GetSparsityPatternCols(full_sparsity, vel_dim_, vel_dim_);
        ws_->sp_int_dvk = ad::CppADInterface::GetSparsityPatternCols(full_sparsity, 2*vel_dim_, vel_dim_);
        ws_->sp_int_dvkp1 = ad::CppADInterface::GetSparsityPatternCols(full_sparsity, 3*vel_dim_, vel_dim_);

        if (node != 0) {
            // dqk pattern
            int col_start = GetDecisionIdx(node, Configuration);
            AddSparsitySet(ws_->sp_int_dqk, row_start, col_start, constraint_triplets_);

            // dvk pattern
            col_start = GetDecisionIdx(node, Velocity);
            AddSparsitySet(ws_->sp_int_dvk, row_start, col_start, constraint_triplets_);
        }


        // dvkp1
        int col_start = GetDecisionIdx(node + 1, Velocity);
        AddSparsitySet(ws_->sp_int_dvkp1, row_start, col_start, constraint_triplets_);

        // q_kp1
        col_start = GetDecisionIdx(node + 1, Configuration);
        AddSparsitySet(ws_->sp_int_dqkp1, row_start, col_start, constraint_triplets_);

    }

    void FullOrderMpc::AddIDPattern(int node, bool full_order) {
        assert(node != nodes_ - 1);

        const int row_start = GetConstraintRow(node, ID);


        const auto sp_full = inverse_dynamics_constraint_->GetJacobianSparsityPatternSet();
        ws_->sp_dq = ad::CppADInterface::GetSparsityPatternCols(sp_full, 0, vel_dim_);
        ws_->sp_dvk = ad::CppADInterface::GetSparsityPatternCols(sp_full, vel_dim_, vel_dim_);
        ws_->sp_dvkp1 = ad::CppADInterface::GetSparsityPatternCols(sp_full, 2*vel_dim_, vel_dim_);
        ws_->sp_tau = ad::CppADInterface::GetSparsityPatternCols(sp_full, 3*vel_dim_, input_dim_);
        ws_->sp_df = ad::CppADInterface::GetSparsityPatternCols(sp_full, 3*vel_dim_ + input_dim_, CONTACT_3DOF*num_contact_locations_);

        ws_->sp_dq_partial.resize(FLOATING_VEL);
        ws_->sp_dvk_partial.resize(FLOATING_VEL);
        ws_->sp_dvkp1_partial.resize(FLOATING_VEL);
        ws_->sp_df_partial.resize(FLOATING_VEL);

        for (int row = 0; row < FLOATING_VEL; row++) {
            ws_->sp_dq_partial[row] = ws_->sp_dq[row];
            ws_->sp_dvk_partial[row] = ws_->sp_dvk[row];
            ws_->sp_dvkp1_partial[row] = ws_->sp_dvkp1[row];

            ws_->sp_df_partial[row] = ws_->sp_df[row];
        }

        // Full order dynamics
        if (full_order) {
            if (node != 0) {
                // dtau_dq
                int col_start = GetDecisionIdx(node, Configuration);
                AddSparsitySet(ws_->sp_dq, row_start, col_start, constraint_triplets_);

                // dtau_dv
                col_start = GetDecisionIdx(node, Velocity);
                AddSparsitySet(ws_->sp_dvk, row_start, col_start, constraint_triplets_);
            }


            // dtau_dtau
            int col_start = GetDecisionIdx(node, Torque);
            AddSparsitySet(ws_->sp_tau, row_start, col_start, constraint_triplets_);

            // dtau_df
            col_start = GetDecisionIdx(node, GroundForce);
            AddSparsitySet(ws_->sp_df, row_start, col_start, constraint_triplets_);

            // dtau_dv2
            col_start = GetDecisionIdx(node + 1, Velocity);
            AddSparsitySet(ws_->sp_dvkp1, row_start, col_start, constraint_triplets_);

        } else {    // Floating base dynamics
            // dtau_dq
            int col_start = GetDecisionIdx(node, Configuration);
            AddSparsitySet(ws_->sp_dq_partial, row_start, col_start, constraint_triplets_);
            // ws_->id_config_mat.setConstant(robot_model_->GetNumInputs() + FLOATING_VEL, robot_model_->GetVelDim(), 1);
            // MatrixToNewTriplet(ws_->id_config_mat.topRows<FLOATING_VEL>(), row_start, col_start, constraint_triplets_);

            // dtau_dv
            col_start = GetDecisionIdx(node, Velocity);
            AddSparsitySet(ws_->sp_dvk_partial, row_start, col_start, constraint_triplets_);
            // ws_->id_vel1_mat.setConstant(robot_model_->GetNumInputs() + FLOATING_VEL, robot_model_->GetVelDim(), 1);
            // MatrixToNewTriplet(ws_->id_vel1_mat.topRows<FLOATING_VEL>(), row_start, col_start, constraint_triplets_);

            // dtau_dtau -- does not appear

            // dtau_df
            col_start = GetDecisionIdx(node, GroundForce);
            AddSparsitySet(ws_->sp_df_partial, row_start, col_start, constraint_triplets_);
            // ws_->id_force_mat.setConstant(robot_model_->GetNumInputs() + FLOATING_VEL, num_contact_locations_*CONTACT_3DOF, 1);
            // MatrixToNewTriplet(ws_->id_force_mat.topRows<FLOATING_VEL>(), row_start, col_start, constraint_triplets_);

            // dtau_dv2
            col_start = GetDecisionIdx(node + 1, Velocity);
            AddSparsitySet(ws_->sp_dvkp1_partial, row_start, col_start, constraint_triplets_);
            // ws_->id_vel2_mat.setConstant(robot_model_->GetNumInputs() + FLOATING_VEL, robot_model_->GetVelDim(), 1);
            // MatrixToNewTriplet(ws_->id_vel2_mat.topRows<FLOATING_VEL>(), row_start, col_start, constraint_triplets_);
        }
    }

    void FullOrderMpc::AddFrictionConePattern(int node) {
        int row_start = GetConstraintRow(node, FrictionCone);
        int col_start = GetDecisionIdx(node, GroundForce);

        matrixx_t id;
        id.setIdentity(CONTACT_3DOF, CONTACT_3DOF);

        // vector3_t h = {1, 0, 0};
        // vector3_t l = {0, 1, 0};
        // vector3_t n = {0, 0, 1};

        const auto sparsity = friction_cone_constraint_->GetJacobianSparsityPatternSet();

        // ws_->fric_cone_mat.resize(FRICTION_CONE_SIZE, CONTACT_3DOF);
        // ws_->fric_cone_mat << (h - n*friction_coef_).transpose(),
        //         -(h + n*friction_coef_).transpose(),
        //         (l - n*friction_coef_).transpose(),
        //         -(l + n*friction_coef_).transpose();
        // for (int i = 0; i < ws_->fric_cone_mat.rows(); i++) {
        //     for (int j = 0; j < ws_->fric_cone_mat.cols(); j++) {
        //         if (ws_->fric_cone_mat(i, j) != 0) {
        //             ws_->fric_cone_mat(i, j) = 1;
        //         }
        //     }
        // }

        for (int contact = 0; contact < num_contact_locations_; contact++) {
            // ----- Zero Force in Swing ----- //
            MatrixToNewTriplet(id, row_start, col_start, constraint_triplets_);
            row_start += CONTACT_3DOF;

            // ----- Friction Cone ----- //
            AddSparsitySet(sparsity, row_start, col_start, constraint_triplets_);
            row_start += friction_cone_constraint_->GetRangeSize();
            col_start += CONTACT_3DOF;

            // ----- Z force bounds ----- //
            // MatrixToNewTriplet(id.topLeftCorner<1,1>(), row_start, col_start - 1, constraint_triplets_);
            // row_start += 1;
        }
    }

    void FullOrderMpc::AddConfigurationBoxPattern(int node) {
        const int row_start = GetConstraintRow(node, ConfigBox);
        const int col_start = GetDecisionIdx(node, Configuration) + FLOATING_VEL;

        // Only have box constraints on the joints
        matrixx_t id = matrixx_t::Identity(robot_model_->GetConfigDim() - FLOATING_BASE,
            robot_model_->GetConfigDim() - FLOATING_BASE);
        MatrixToNewTriplet(id, row_start, col_start, constraint_triplets_);
    }

    void FullOrderMpc::AddVelocityBoxPattern(int node) {
        const int row_start = GetConstraintRow(node, VelBox);
        const int col_start = GetDecisionIdx(node, Velocity) + FLOATING_VEL;

        matrixx_t id;
        // Only apply box constraints on joints, not the floating base
        id.setIdentity(robot_model_->GetVelDim() - FLOATING_VEL, robot_model_->GetVelDim() - FLOATING_VEL);
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

        for (const auto& frame : contact_frames_) {
            const auto sparsity = swing_height_constraint_[frame]->GetJacobianSparsityPatternSet();

            // dqk
            AddSparsitySet(sparsity, row_start, col_start, constraint_triplets_);

            row_start++;
        }
    }

    void FullOrderMpc::AddHolonomicPattern(int node) {
        int row_start = GetConstraintRow(node, Holonomic);

        // TODO: Remove
        ws_->holo_mat.setConstant(2, robot_model_->GetVelDim(), 1);

        for (const auto& frame : contact_frames_) {
            const auto full_sparsity = holonomic_constraint_[frame]->GetJacobianSparsityPatternSet();
            ws_->sp_hol_dqk[frame] = ad::CppADInterface::GetSparsityPatternCols(full_sparsity, 0, vel_dim_);
            ws_->sp_hol_dvk[frame] = ad::CppADInterface::GetSparsityPatternCols(full_sparsity, vel_dim_, vel_dim_);

            int col_start = GetDecisionIdx(node, Configuration);
            AddSparsitySet(ws_->sp_hol_dqk[frame], row_start, col_start, constraint_triplets_);

            col_start = GetDecisionIdx(node, Velocity);
            AddSparsitySet(ws_->sp_hol_dvk[frame], row_start, col_start, constraint_triplets_);

            row_start += 2;
        }
    }

    void FullOrderMpc::AddCollisionPattern(int node) {
        int row_start = GetConstraintRow(node, Collision);
        const int col_start = GetDecisionIdx(node, Configuration);

        for (const auto& constraint : collision_constraints_) {
            const auto sparsity = constraint->GetJacobianSparsityPatternSet();
            AddSparsitySet(sparsity, row_start, col_start, constraint_triplets_);
            row_start++;
        }
    }


    // ------------------------------------------------- //
    // ---------------- Helper Functions --------------- //
    // ------------------------------------------------- //
    void FullOrderMpc::ConvertSolutionToTraj(const vectorx_t& qp_sol, Trajectory& traj) {
        if (traj.GetNumNodes() != nodes_) {
            traj.SetNumNodes(nodes_);
        }

        vectorx_t config_new;
        for (int node = 0; node < nodes_; node++) {
            if (node != 0) {
                ConvertdqToq(qp_sol.segment(GetDecisionIdx(node, Configuration), robot_model_->GetVelDim()), traj_.GetConfiguration(node), config_new);
                traj.SetVelocity(node,
                    qp_sol.segment(GetDecisionIdx(node, Velocity), robot_model_->GetVelDim()) + traj_.GetVelocity(node));
            } else {
                config_new = traj_.GetConfiguration(0);
                traj.SetVelocity(node, traj_.GetVelocity(0));
            }

            traj.SetConfiguration(node, config_new);


            if (node < nodes_full_dynamics_) {
                traj.SetTau(node,
                    qp_sol.segment(GetDecisionIdx(node, Torque), robot_model_->GetNumInputs()) + traj_.GetTau(node));
            } else {
                traj.SetTau(node, traj_.GetTau(node));
            }
            int force_idx = 0;
            for (const auto& frame : contact_frames_) {
                traj.SetForce(node, frame,
                    qp_sol.segment<3>(GetDecisionIdx(node, GroundForce) + 3*force_idx) + traj_.GetForce(node, frame));
                force_idx++;
            }
        }
    }


    int FullOrderMpc::GetNumConstraints() const {
        return GetConstraintRowStartNode(nodes_);
    }

    int FullOrderMpc::GetNumDecisionVars() const {
        return GetDecisionIdxStart(nodes_);
    }

    int FullOrderMpc::GetDecisionIdxStart(int node) const {
        const int num_inputs = robot_model_->GetNumInputs();

        int idx = 0;
        for (int i = 0; i < node; i++) {
            if (i < nodes_full_dynamics_) {
                idx += num_inputs;
            }

            if (i != 0) {
                idx += 2*vel_dim_;
            }

            idx += num_contact_locations_*CONTACT_3DOF;
        }

        return idx;
    }


    int FullOrderMpc::GetDecisionIdx(int node, const DecisionType& var_type) const {
        if (var_type == Torque && node >= nodes_full_dynamics_) {
            throw std::logic_error("There is no torque variable at this node!");
        }

        if (var_type == Configuration && node == 0) {
            throw std::logic_error("There is no configuration variable at node 0!");
        }

        if (var_type == Velocity && node == 0) {
            throw std::logic_error("There is no velocity variable at node 0!");
        }

        int idx = GetDecisionIdxStart(node);
        switch (var_type) {
            case GroundForce:
                if (node < nodes_full_dynamics_) {
                    idx += robot_model_->GetNumInputs();
                }
            case Torque:
                if (node > 0) {
                    idx += vel_dim_;
                }
            case Velocity:
                if (node > 0) {
                    idx += vel_dim_;
                }
            case Configuration:
                break;
        }
        return idx;
    }

    int FullOrderMpc::GetConstraintRowStartNode(int node) const {
        // int row = 2*robot_model_->GetVelDim(); // Initial condition constraint
        int row = 0;
        for (int i = 0; i < node; i++) {
            if (i < nodes_ - 1) {
                row += NumIntegratorConstraintsNode();
            }

            if (i < nodes_full_dynamics_) {
                row += NumIDConstraintsNode();
                row += NumTorqueBoxConstraintsNode();
            } else if (i < nodes_ - 1) {
                row += NumPartialIDConstraintsNode();
            }

            row += NumFrictionConeConstraintsNode();

            if (i > 0) {
                row += NumHolonomicConstraintsNode();
                row += NumVelocityBoxConstraintsNode();

                row += NumConfigBoxConstraintsNode();
                row += NumSwingHeightConstraintsNode();

                row += NumCollisionConstraintsNode();
            }
        }

        return row;
    }


    int FullOrderMpc::GetConstraintRow(int node, const ConstraintType& constraint) const {
        int row = GetConstraintRowStartNode(node);
        switch (constraint) {
            case Collision:
                if (node > 0) {
                    row += NumHolonomicConstraintsNode();
                }
            case Holonomic:
                if (node > 0) {
                    row += NumSwingHeightConstraintsNode();
                }
            case SwingHeight:
                if (node < nodes_full_dynamics_) {
                    row += NumTorqueBoxConstraintsNode();
                }
            case TorqueBox:
                if (node > 0) {
                    row += NumVelocityBoxConstraintsNode();
                }
            case VelBox:
                if (node > 0) {
                    row += NumConfigBoxConstraintsNode();
                }
            case ConfigBox:
                row += NumFrictionConeConstraintsNode();
            case FrictionCone:
                if (node < nodes_full_dynamics_) {
                    row += NumIDConstraintsNode();
                } else if (node < nodes_ - 1) {
                    row += NumPartialIDConstraintsNode();
                }
            case ID:
                if (node < nodes_ - 1) {
                    row += NumIntegratorConstraintsNode();
                }
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

    void FullOrderMpc::AddSparsitySet(const torc::ad::sparsity_pattern_t& sparsity, int row_start, int col_start,
                                      std::vector<Eigen::Triplet<double>>& triplet) {
        for (int row = 0; row < sparsity.size(); row++) {
            for (const auto& col : sparsity[row]) {
                if (row == col){
                    triplet.emplace_back(row_start + row, col_start + col, 100);    // To ensure the sparsity pattern is psd

                } else {
                    triplet.emplace_back(row_start + row, col_start + col, 1);
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

    void FullOrderMpc::MatrixToTripletWithSparsitySet(const torc::mpc::matrixx_t& mat, int row_start, int col_start,
                                                      std::vector<Eigen::Triplet<double>>& triplet, int& triplet_idx,
                                                      const torc::ad::sparsity_pattern_t& sparsity) {
        for (int row = 0; row < sparsity.size(); row++) {
            for (const auto& col : sparsity[row]) {
                triplet[triplet_idx] = Eigen::Triplet<double>(row_start + row, col_start + col, mat(row, col));
                triplet_idx++;
            }
        }
    }

    void FullOrderMpc::DiagonalScalarMatrixToTriplet(double val, int row_start, int col_start, int size, std::vector<Eigen::Triplet<double>>& triplet, int& triplet_idx) {
        for (int idx = 0; idx < size; idx++) {
            triplet[triplet_idx] = Eigen::Triplet<double>(row_start + idx, col_start + idx, val);
            triplet_idx++;
        }
    }

    // TODO: Merge with the one in pinocchio_interface.h
    void FullOrderMpc::ConvertdqToq(const vectorx_t& dq, const vectorx_t& q_ref, vectorx_t& q) const {
        q.resize(robot_model_->GetConfigDim());
        q = q_ref;

        // Position and joints are simple addition
        q.head<POS_VARS>() += dq.head<3>();
        q.tail(robot_model_->GetNumInputs()) += dq.tail(robot_model_->GetNumInputs());

        // Quaternion
        q.segment<QUAT_VARS>(POS_VARS) = (static_cast<quat_t>(q.segment<QUAT_VARS>(POS_VARS))
            * pinocchio::quaternion::exp3(dq.segment<3>(POS_VARS))).coeffs();
    }

    double FullOrderMpc::GetTime(int node) const {
        double time = 0;
        for (int i = 0; i < node; i++) {
            time += dt_[i];
        }

        return time;
    }



    // ------------------------------------------------- //
    // ----- Getters for Sizes of Individual nodes ----- //
    // ------------------------------------------------- //
    int FullOrderMpc::NumIntegratorConstraintsNode() const {
        return vel_dim_;
    }

    int FullOrderMpc::NumIDConstraintsNode() const {
        // Floating base plus the number of inputs
        return FLOATING_VEL + robot_model_->GetNumInputs();
    }

    int FullOrderMpc::NumPartialIDConstraintsNode() const {
        return FLOATING_VEL;
    }


    int FullOrderMpc::NumFrictionConeConstraintsNode() const {
        return num_contact_locations_ * (1 + CONTACT_3DOF + 1);
    }

    int FullOrderMpc::NumConfigBoxConstraintsNode() const {
        return robot_model_->GetConfigDim() - FLOATING_BASE;
    }

    int FullOrderMpc::NumVelocityBoxConstraintsNode() const {
        return robot_model_->GetVelDim() - FLOATING_VEL;
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

    int FullOrderMpc::NumCollisionConstraintsNode() const {
        return collision_constraints_.size();
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
        traj_.SetDtVector(dt_);
    }

    void FullOrderMpc::SetConstantConfigTarget(const vectorx_t& q_target) {
        std::lock_guard<std::mutex> lock(target_mut_);

        if (q_target.size() != config_dim_) {
            throw std::runtime_error("Configuration target does not have the correct size!");
        }

        q_target_ = SimpleTrajectory(config_dim_, nodes_);
        q_target_->SetAllData(q_target);
    }

    void FullOrderMpc::SetConstantVelTarget(const vectorx_t& v_target) {
        std::lock_guard<std::mutex> lock(target_mut_);

        if (v_target.size() != vel_dim_) {
            throw std::runtime_error("Velocity target does not have the correct size!");
        }

        v_target_ = SimpleTrajectory(vel_dim_, nodes_);

        v_target_->SetAllData(v_target);
    }

    void FullOrderMpc::SetConfigTarget(SimpleTrajectory q_target) {
        std::lock_guard<std::mutex> lock(target_mut_);

        if (q_target.GetNumNodes() != nodes_) {
            throw std::invalid_argument("Configuration target has the wrong number of nodes!");
        }

        if (q_target.GetSize() != config_dim_) {
            throw std::runtime_error("Configuration target does not have the correct size!");
        }

        for (int node = 0; node < nodes_; node++) {
            q_target[node].segment<QUAT_VARS>(POS_VARS).normalize();
        }

        q_target_ = std::move(q_target);
    }

    void FullOrderMpc::SetVelTarget(SimpleTrajectory v_target) {
        std::lock_guard<std::mutex> lock(target_mut_);

        if (v_target.GetNumNodes() != nodes_) {
            throw std::invalid_argument("Velocity target has the wrong number of nodes!");
        }

        if (v_target.GetSize() != vel_dim_) {
            throw std::runtime_error("Velocity target does not have the correct size!");
        }

        v_target_ = std::move(v_target);
    }

    void FullOrderMpc::SetSwingFootTrajectory(const std::string& frame, const std::vector<double>& swing_traj) {
        if (swing_traj_.find(frame) == swing_traj_.end()) {
            throw std::invalid_argument("Swing frame not found!");
        }

        if (swing_traj.size() != nodes_) {
            throw std::invalid_argument("Swing trajectory does not have the correct number of nodes!");
        }

        swing_traj_[frame] = swing_traj;
    }

    bool FullOrderMpc::PlannedContact(const std::string& frame, int node) const {
        try {
            int in_contact = in_contact_.at(frame).at(node);
            if (in_contact != 0) {
                return true;
            }
            return false;
        } catch (const std::out_of_range& e) {
            throw std::runtime_error("Frame or node not found!");
        }
    }


    void FullOrderMpc::PrintStatistics() const {
        using std::setw;
        using std::setfill;

        const int col_width = 25;
        const int total_width = 12*col_width;


        auto time_now = std::chrono::system_clock::now();
        std::time_t time1_now = std::chrono::system_clock::to_time_t(time_now);

        std::cout << setfill('=') << setw(total_width/2 - 7) << "" << " MPC Statistics " << setw(total_width/2 - 7) << "" << std::endl;
        std::cout << setfill(' ');
        std::cout << setw(col_width) << "Solve #"
                << setw(col_width) << "Solve Status"
                << setw(col_width) << "Time (ms)"
                << setw(col_width) << "Constr. Time (ms)"
                << setw(col_width) << "Cost Time (ms)"
                << setw(col_width) << "LS Time (ms)"
                << setw(col_width) << "d-norm (post LS)"
                << setw(col_width) << "Alpha"
                << setw(col_width) << "LS Termination"
                << setw(col_width) << "Constr. Vio. (Scaled)"
                << setw(col_width) << "Cost (post LS)"
                << setw(col_width) << "QP Cost (pre LS)" << std::endl;
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
                    solve_status = "Primal Infeas. Inacc.";
                    break;
                case osqp::OsqpExitCode::kDualInfeasibleInaccurate:
                    solve_status = "Dual Infeas. Inacc.";
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

            std::string ls_termination_condition;
            switch (stats_[solve].ls_condition) {
                case ConstraintViolation:
                    ls_termination_condition = "Constraint Vio.";
                    break;
                case CostReduction:
                    ls_termination_condition = "Cost Red.";
                    break;
                case Both:
                    ls_termination_condition = "Cost & Constr.";
                    break;
                case MinAlpha:
                    ls_termination_condition = "Alpha small";
                    break;
            }
            std::cout << setw(col_width) << solve
                      << setw(col_width) << solve_status
                      << setw(col_width) << stats_[solve].total_compute_time
                      << setw(col_width) << stats_[solve].constraint_time
                      << setw(col_width) << stats_[solve].cost_time
                      << setw(col_width) << stats_[solve].ls_time
                      << setw(col_width) << stats_[solve].qp_res_norm
                      << setw(col_width) << stats_[solve].alpha
                      << setw(col_width) << ls_termination_condition
                      << setw(col_width) << stats_[solve].constraint_violation
                      << setw(col_width) << stats_[solve].full_cost
                      << setw(col_width) << stats_[solve].qp_cost << std::endl;
        }
    }

    void FullOrderMpc::PrintContactSchedule() const {
        std::cout << std::setfill(' ');
        std::cout << std::setw(22) << "";
        double time = 0.00;
        for (int node = 0; node < nodes_; node++) {
            std::cout << std::setw(1) << std::fixed << std::setprecision(2) << " " << time;

            time += dt_[node];
        }

        for (const auto& frame : contact_frames_) {
            std::cout << std::endl;

            std::cout << std::setw(20) << frame << "  ";
            for (int node = 0; node < nodes_; node++) {
                if (in_contact_.at(frame)[node]) {
                    std::cout << " ____";
                } else {
                    std::cout << " xxxx";
                }
            }
            std::cout << std::endl;

            // Reset precision and formatting to default
            std::cout.unsetf(std::ios::fixed);
            std::cout.precision(6); // Default precision (usually 6)
        }
    }

    void FullOrderMpc::PrintAggregateStats() const {
        using std::setw;
        using std::setfill;

        const int col_width = 25;
        const int total_width = 3*col_width;


        auto time_now = std::chrono::system_clock::now();
        std::time_t time1_now = std::chrono::system_clock::to_time_t(time_now);

        std::cout << setfill('=') << setw(total_width/2 - 10) << "" << " MPC Aggregate Statistics " << setw(total_width/2 - 11) << "" << std::endl;
        std::cout << setfill(' ');

        std::cout << std::left << setw(15) << "Total solves: " << std::right << total_solves_ << std::endl;
        std::cout << std::left << setw(col_width) << "" << std::right << "Average" << std::right << std::setw(col_width + 9) << "Standard deviation" << std::endl;
        std::cout << setfill('-') << setw(3*col_width) << "" << setfill(' ') << std::endl;
        const auto compute_times = GetComputeTimeStats();
        const auto constraint_times = GetConstraintTimeStats();
        const auto cost_times = GetCostTimeStats();
        const auto ls_times = GetLineSearchTimeStats();

        std::cout << std::fixed << std::setprecision(6);
        std::cout << std::left << setw(col_width) << "Compute time (ms): " << std::right << compute_times.first <<
            setw(col_width) << compute_times.second << std::endl;
        std::cout << std::left << setw(col_width) << "Constraint time (ms): " << std::right << constraint_times.first <<
            setw(col_width) << constraint_times.second << std::endl;
        std::cout << std::left << setw(col_width) << "Cost time (ms): " << std::right << cost_times.first <<
            setw(col_width) << cost_times.second << std::endl;
        std::cout << std::left << setw(col_width) << "Line search time (ms): " << std::right << ls_times.first <<
            setw(col_width) << ls_times.second << std::endl;

        const auto constr_vio_stats = GetConstraintViolationStats();
        std::cout << std::left << setw(col_width) << "Constraint violation: " << std::right << constr_vio_stats.first <<
            setw(col_width) << constr_vio_stats.second << std::endl;

        const auto cost_stats = GetCostStats();
        std::cout << std::left << setw(col_width) << "Cost: " << std::right << cost_stats.first <<
            setw(col_width) << cost_stats.second << std::endl;
    }

    std::pair<double, double> FullOrderMpc::GetComputeTimeStats() const {
        double avg = 0;
        for (const auto& stat : stats_) {
            avg += stat.total_compute_time;
        }

        avg = avg/stats_.size();

        double st_dev = 0;
        for (const auto& stat : stats_) {
            st_dev += std::pow((stat.total_compute_time - avg), 2);
        }

        st_dev = sqrt(st_dev/stats_.size());

        return std::make_pair(avg, st_dev);
    }

    std::pair<double, double> FullOrderMpc::GetConstraintTimeStats() const {
        double avg = 0;
        for (const auto& stat : stats_) {
            avg += stat.constraint_time;
        }

        avg = avg/stats_.size();

        double st_dev = 0;
        for (const auto& stat : stats_) {
            st_dev += std::pow((stat.constraint_time - avg), 2);
        }

        st_dev = sqrt(st_dev/stats_.size());

        return std::make_pair(avg, st_dev);
    }

    std::pair<double, double> FullOrderMpc::GetCostTimeStats() const {
        double avg = 0;
        for (const auto& stat : stats_) {
            avg += stat.cost_time;
        }

        avg = avg/stats_.size();

        double st_dev = 0;
        for (const auto& stat : stats_) {
            st_dev += std::pow((stat.cost_time - avg), 2);
        }

        st_dev = sqrt(st_dev/stats_.size());

        return std::make_pair(avg, st_dev);
    }

    std::pair<double, double> FullOrderMpc::GetLineSearchTimeStats() const {
        double avg = 0;
        for (const auto& stat : stats_) {
            avg += stat.ls_time;
        }

        avg = avg/stats_.size();

        double st_dev = 0;
        for (const auto& stat : stats_) {
            st_dev += std::pow((stat.ls_time - avg), 2);
        }

        st_dev = sqrt(st_dev/stats_.size());

        return std::make_pair(avg, st_dev);
    }

    std::pair<double, double> FullOrderMpc::GetConstraintViolationStats() const {
        double avg = 0;
        for (const auto& stat : stats_) {
            avg += stat.constraint_violation;
        }

        avg = avg/stats_.size();

        double st_dev = 0;
        for (const auto& stat : stats_) {
            st_dev += std::pow((stat.constraint_violation - avg), 2);
        }

        st_dev = sqrt(st_dev/stats_.size());

        return std::make_pair(avg, st_dev);
    }

    std::pair<double, double> FullOrderMpc::GetCostStats() const {
        double avg = 0;
        for (const auto& stat : stats_) {
            avg += stat.full_cost;
        }

        avg = avg/stats_.size();

        double st_dev = 0;
        for (const auto& stat : stats_) {
            st_dev += std::pow((stat.full_cost - avg), 2);
        }

        st_dev = sqrt(st_dev/stats_.size());

        return std::make_pair(avg, st_dev);
    }

    long FullOrderMpc::GetTotalSolves() const {
        return total_solves_;
    }

    const std::vector<double>& FullOrderMpc::GetDtVector() const {
        return dt_;
    }

    void FullOrderMpc::PrintSwingTraj(const std::string& frame) const {
        std::cout << "Frame: " << frame << std::endl;

        double time = 0;
        std::cout << "Time: ";
        for (int node = 0; node < nodes_; node++) {
            std::cout << std::setw(1) << std::fixed << std::setprecision(3) << " " << time;

            time += dt_[node];
        }

        std::cout << std::endl << "      ";
        for (int node = 0; node < nodes_; node++) {
            std::cout << std::setw(1) << std::fixed << std::setprecision(3) << " " << swing_traj_.at(frame)[node];
        }
        std::cout << std::setprecision(6) << std::endl;
    }


} // namespace torc::mpc
