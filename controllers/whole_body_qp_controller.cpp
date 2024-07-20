//
// Created by zolkin on 7/17/24.
//

#include "whole_body_qp_controller.h"
#include "yaml-cpp/yaml.h"

namespace torc::controllers {
    WholeBodyQPController::WholeBodyQPController(const std::string& name)
        : name_(name) {}

    WholeBodyQPController::WholeBodyQPController(const std::string& name, const models::FullOrderRigidBody& model,
                                                 const fs::path& config_file_path)
        : WholeBodyQPController(name) {
        config_file_path_ = config_file_path;
        ParseUpdateSettings();

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

        // ---------- Solver Settings ---------- //
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

        // ---------- Constraint Settings ---------- //
        if (!config["constraints"]) {
            throw std::runtime_error("No constraint settings provided!");
        }
        YAML::Node constraint_settings = config["constraints"];
        auto torque_bound = constraint_settings["torque_bounds"].as<std::vector<double>>();
        ConvertStdVectorToEigen(torque_bound, torque_bounds_);
        friction_coef_ = constraint_settings["friction_coef"].as<double>();
        max_grf_ = constraint_settings["max_grf"].as<double>();

        // ---------- Cost Settings ---------- //
        if (!config["costs"]) {
            throw std::runtime_error("No cost settings provided!");
        }
        YAML::Node cost_settings = config["costs"];
        leg_tracking_weight_ = cost_settings["leg_tracking_weight"].as<double>();
        torso_tracking_weight_ = cost_settings["torso_tracking_weight"].as<double>();
        force_tracking_weight_ = cost_settings["force_tracking_weight"].as<double>();
        auto kd_joint = cost_settings["kd_joint"].as<std::vector<double>>();
        ConvertStdVectorToEigen(kd_joint, kd_joint_);
        auto kp_joint = cost_settings["kp_joint"].as<std::vector<double>>();
        ConvertStdVectorToEigen(kp_joint, kp_joint_);
        kp_pos_ = cost_settings["kp_base_pos"].as<double>();
        kp_orientation_ = cost_settings["kp_base_orientation"].as<double>();
        kd_pos_ = cost_settings["kd_base_pos"].as<double>();
        kd_orientation_ = cost_settings["kd_base_orientation"].as<double>();
    }

    vectorx_t WholeBodyQPController::ComputeControl(const vectorx_t& target_state, const vectorx_t& force_target,
                                                    const vectorx_t& state, const models::RobotContactInfo& contact) {
        model_->ParseState(target_state, q_target_, v_target_);
        force_target_ = force_target;

        // For now, only work with point contacts
        num_contacts_ = 0;
        for (const auto& con : contact.contacts) {
            if (con.second.type != models::PointContact) {
                throw std::runtime_error("[WBC] Only point contacts are currently supported!");
            }

            if (con.second.state) {
                num_contacts_++;
            }
        }

        // --------- Constraints --------- //
        // Resize constraints if necessary
        long num_decision_vars = model_->GetVelDim() + POINT_CONTACT_CONSTRAINTS*num_contacts_;
        long num_constraints = FLOATING_VEL_OFFSET + 8*num_contacts_ + model_->GetNumInputs();
        constraints_.SetSizes(num_constraints, num_decision_vars);
        constraints_.Zero();

        // Get info from the model
        model_->GetDynamicsTerms(state, M_, C_, g_);
        vectorx_t q, v;
        model_->ParseState(state, q, v);

        // Dynamics constraints
        AddDynamicsConstraints(q, v, contact);

        // Holonomic constraints
        AddHolonomicConstraints(q, v, contact);

        // Torque constraints
        AddTorqueConstraints(q, v, contact);

        // Friction cone constraints
        AddFrictionConeConstraints(contact);

        // Positive GRF constraints
        AddPositiveGRFConstraints(contact);

        std::cout << "A: " << constraints_.A << std::endl;
        std::cout << "ub: " << constraints_.ub << std::endl;
        std::cout << "lb: " << constraints_.lb << std::endl;

        // --------- Costs --------- //
        P_.resize(num_decision_vars, num_decision_vars);
        P_.setZero();
        w_.resize(num_decision_vars);
        w_.setZero();

        // Leg tracking costs
        AddLegTrackingCost(q, v);

        // Torso tracking costs
        AddTorsoTrackingCost(q, v);

        // Force tracking costs
        AddForceTrackingCost(q, v);

        std::cout << "P: " << P_ << std::endl;
        std::cout << "w: " << w_ << std::endl;

        // --------- Update QP & Solve --------- //

        return vectorx_t::Zero(1);
    }

    // ---------------------------------------- //
    // --------- Constraint Functions --------- //
    // ---------------------------------------- //
    void WholeBodyQPController::AddDynamicsConstraints(const vectorx_t& q, const vectorx_t& v,
                                                       const models::RobotContactInfo& contact) {

        constraints_.lb.head<FLOATING_VEL_OFFSET>() = -(g_.head(FLOATING_VEL_OFFSET) +
                                                        (C_*v).head(FLOATING_VEL_OFFSET));
        constraints_.ub.head<FLOATING_VEL_OFFSET>() = constraints_.lb.head<FLOATING_VEL_OFFSET>();

        constraints_.A.topLeftCorner(FLOATING_VEL_OFFSET, model_->GetVelDim()) =
                M_.topLeftCorner(FLOATING_VEL_OFFSET, model_->GetVelDim());

        int contact_num = 0;
        for (const auto& con : contact.contacts) {
            if (con.second.state) {
                matrixx_t J = matrixx_t::Zero(FLOATING_VEL_OFFSET, model_->GetVelDim());
                model_->GetFrameJacobian(con.first, q, J);
                constraints_.A.block(0, model_->GetVelDim() + contact_num*POINT_CONTACT_CONSTRAINTS,
                         FLOATING_VEL_OFFSET, POINT_CONTACT_CONSTRAINTS) = -J.topRows<POINT_CONTACT_CONSTRAINTS>().transpose().topRows<FLOATING_VEL_OFFSET>();
                contact_num++;
            }
        }
    }

    void WholeBodyQPController::AddHolonomicConstraints(const vectorx_t& q, const vectorx_t& v,
                                                        const models::RobotContactInfo& contact) {
        if (num_contacts_ > 0) {
            int contact_idx = 0;
            model_->ThirdOrderFK(q, v, 0*v);
            for (const auto& con : contact.contacts) {
                if (con.second.state) {
                    constraints_.lb.segment<3>(HolonomicIdx() + contact_idx*3) = -model_->GetFrameAcceleration(con.first).linear();
                    constraints_.ub.segment<3>(HolonomicIdx() + contact_idx*3) = constraints_.lb.segment<3>(FLOATING_VEL_OFFSET + contact_idx*3);

                    matrixx_t J = matrixx_t::Zero(FLOATING_VEL_OFFSET, model_->GetVelDim());
                    model_->GetFrameJacobian(con.first, q, J);
                    constraints_.A.block(HolonomicIdx() + contact_idx*POINT_CONTACT_CONSTRAINTS,
                                         0, POINT_CONTACT_CONSTRAINTS, model_->GetVelDim()) = J.topRows<POINT_CONTACT_CONSTRAINTS>();
                    contact_idx++;
                }
            }

        }
    }

    void WholeBodyQPController::AddTorqueConstraints(const torc::controllers::vectorx_t& q, const torc::controllers::vectorx_t& v,
                                                     const models::RobotContactInfo& contact) {
        constraints_.A.block(TorqueIdx(), 0, model_->GetNumInputs(), model_->GetVelDim()) = M_.bottomRows(model_->GetNumInputs());
        if (num_contacts_ > 0) {
            int contact_idx = 0;
            for (const auto& con : contact.contacts) {
                if (con.second.state) {
                    matrixx_t J = matrixx_t::Zero(FLOATING_VEL_OFFSET, model_->GetVelDim());
                    model_->GetFrameJacobian(con.first, q, J);
                    constraints_.A.block(TorqueIdx(),  model_->GetVelDim() + contact_idx*POINT_CONTACT_CONSTRAINTS, model_->GetNumInputs(),
                                         POINT_CONTACT_CONSTRAINTS) = -J.topRows<POINT_CONTACT_CONSTRAINTS>().transpose().bottomRows(model_->GetNumInputs());
                    contact_idx++;
                }
            }
        }

        constraints_.lb.segment(TorqueIdx(), model_->GetNumInputs()) = -(C_*v + g_).segment(FLOATING_VEL_OFFSET, model_->GetNumInputs()) - torque_bounds_;
        constraints_.ub.segment(TorqueIdx(), model_->GetNumInputs()) = -(C_*v + g_).segment(FLOATING_VEL_OFFSET, model_->GetNumInputs()) + torque_bounds_;
    }

    void WholeBodyQPController::AddFrictionConeConstraints(const models::RobotContactInfo& contact) {
        Eigen::Vector3d h = {1, 0, 0};
        Eigen::Vector3d l = {0, 1, 0};
        Eigen::Vector3d n = {0, 0, 1};

        for (int i = 0; i < num_contacts_; i++) {
            constraints_.A.block(FrictionIdx() + 4 * i,
                                 model_->GetVelDim() + 3 * i, 4, 3) <<
                                                         (h - n * friction_coef_).transpose(),
                    -(h + n * friction_coef_).transpose(),
                    (l - n * friction_coef_).transpose(),
                    -(l + n * friction_coef_).transpose();
            constraints_.ub.segment(FrictionIdx() + 4 * i, 4) =
                    Eigen::Vector4d::Zero();
            constraints_.lb.segment(FrictionIdx() + 4 * i, 4) =
                    Eigen::Vector4d::Ones() * -OsqpEigen::INFTY;
        }
    }

    void WholeBodyQPController::AddPositiveGRFConstraints(const models::RobotContactInfo& contact) {
        for (int i = 0; i < num_contacts_; i++) {
            constraints_.A(PosGRFIdx() + i,
               model_->GetVelDim() + 2 + 3*i) = 1;
            constraints_.ub(PosGRFIdx() + i) = max_grf_;
            constraints_.lb(PosGRFIdx() + i) = 0;
        }
    }

    // ---------------------------------------- //
    // ------------ Cost Functions ------------ //
    // ---------------------------------------- //
    void WholeBodyQPController::AddLegTrackingCost(const vectorx_t& q, const vectorx_t& v) {
        const size_t num_inputs = model_->GetNumInputs();
        P_.block(FLOATING_VEL_OFFSET, FLOATING_VEL_OFFSET, num_inputs, num_inputs) =
                leg_tracking_weight_ * 2*Eigen::MatrixXd::Identity(num_inputs, num_inputs);

        // For now, just make all the joint accelerations go to 0 - so the term does not appear here
        Eigen::VectorXd target = kd_joint_.cwiseProduct(v_target_.tail(num_inputs) - v.tail(num_inputs)) +
                                 kp_joint_.cwiseProduct(q_target_.tail(num_inputs) - q.tail(num_inputs));

        w_.segment(FLOATING_VEL_OFFSET, num_inputs) = -2*target*leg_tracking_weight_;
    }

    void WholeBodyQPController::AddTorsoTrackingCost(const torc::controllers::vectorx_t& q,
                                                     const torc::controllers::vectorx_t& v) {
        // Add the position costs
        P_.topLeftCorner(POS_VARS, POS_VARS) =
                torso_tracking_weight_*2*Eigen::MatrixXd::Identity(POS_VARS, POS_VARS);

        // For now, just make all the joint accelerations go to 0 - so the term does not appear here
        Eigen::VectorXd target = kd_pos_ * (v_target_.head(POS_VARS) - v.head<POS_VARS>())
                                 + kp_pos_*(q_target_.head(POS_VARS) - q.head<POS_VARS>());

        w_.head(POS_VARS) = -2*target*torso_tracking_weight_;

        // Add the orientation costs
        // TODO: Orientation costs break the system
        P_.block(POS_VARS, POS_VARS, 3, 3) =
                torso_tracking_weight_*2*Eigen::MatrixXd::Identity(3,3);

        Eigen::Quaternion<double> orientation(static_cast<Eigen::Vector4d>(q.segment(POS_VARS, 4)));
        orientation.normalize();
        Eigen::Quaternion<double> des_orientation(static_cast<Eigen::Vector4d>(q_target_.segment(POS_VARS, 4)));
        des_orientation.normalize();

        // Note: velocity orientations are in different frames. Need to convert.
        // Should be able to exp the vel back to the surface. Then invert to the local frame then log back to the tangent space
        Eigen::Quaterniond vel_quat;
        Eigen::Vector3d temp = v_target_.segment<3>(POS_VARS);
        pinocchio::quaternion::exp3(temp, vel_quat);
        vel_quat = orientation.inverse()*vel_quat;
        Eigen::Vector3d vel_frame = pinocchio::quaternion::log3(vel_quat);

        Eigen::VectorXd angle_target = kd_orientation_*(vel_frame - v.segment(POS_VARS, 3)) +
                                       kp_orientation_*pinocchio::quaternion::log3(orientation.inverse()*des_orientation);

        w_.segment(POS_VARS, 3) = -2*angle_target*torso_tracking_weight_;
    }

    void WholeBodyQPController::AddForceTrackingCost(const torc::controllers::vectorx_t& q,
                                                     const torc::controllers::vectorx_t& v) {
        P_.block(FLOATING_VEL_OFFSET + model_->GetNumInputs(), model_->GetVelDim(),
                 POINT_CONTACT_CONSTRAINTS*num_contacts_, POINT_CONTACT_CONSTRAINTS*num_contacts_) =
                force_tracking_weight_*2*Eigen::MatrixXd::Identity(POINT_CONTACT_CONSTRAINTS*num_contacts_, POINT_CONTACT_CONSTRAINTS*num_contacts_);

        Eigen::VectorXd target = force_target_;

        w_.segment(FLOATING_VEL_OFFSET + model_->GetNumInputs(), POINT_CONTACT_CONSTRAINTS*num_contacts_) = -2*target*force_tracking_weight_;
    }

    // ---------------------------------------- //
    // ----------- Helper Functions ----------- //
    // ---------------------------------------- //
    int WholeBodyQPController::DynamicsIdx() const {
        return 0;
    }

    int WholeBodyQPController::HolonomicIdx() const {
        return FLOATING_VEL_OFFSET;
    }

    int WholeBodyQPController::TorqueIdx() const {
        return FLOATING_VEL_OFFSET + POINT_CONTACT_CONSTRAINTS*num_contacts_;
    }

    int WholeBodyQPController::FrictionIdx() const {
        return FLOATING_VEL_OFFSET + POINT_CONTACT_CONSTRAINTS*num_contacts_ + model_->GetNumInputs();
    }

    int WholeBodyQPController::PosGRFIdx() const {
        return FLOATING_VEL_OFFSET + POINT_CONTACT_CONSTRAINTS*num_contacts_ + model_->GetNumInputs() + 4*num_contacts_;
    }

    void WholeBodyQPController::ConvertStdVectorToEigen(const std::vector<double>& v1,
                                                        torc::controllers::vectorx_t& v2) {
        v2.resize(v1.size());
        for (int i = 0; i < v1.size(); i++) {
            v2(i) = v1[i];
        }
    }
}