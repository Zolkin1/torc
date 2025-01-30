//
// Created by zolkin on 1/18/25.
//

#include "pinocchio_model.h"
#include "DynamicsConstraint.h"

#include <pinocchio/algorithm/joint-configuration.hpp>

#include "pinocchio_interface.h"



namespace torc::mpc {
    DynamicsConstraint::DynamicsConstraint(const models::FullOrderRigidBody& model,
        const std::vector<std::string>& contact_frames, const std::string& name, const fs::path& deriv_lib_path,
        bool compile_derivs, bool full_order,
        int first_node, int last_node)
        : Constraint(first_node, last_node, name), full_order_(full_order), model_(model) {

        vel_dim_ = model_.GetVelDim();
        config_dim_ = model_.GetConfigDim();
        tau_dim_ = model_.GetVelDim() - FLOATING_VEL;
        num_contacts_ = contact_frames.size();
        contact_frames_ = contact_frames;

        // Make the auto diff function
        dynamics_function_ = std::make_unique<ad::CppADInterface>(
            std::bind(&DynamicsConstraint::InverseDynamics, this, contact_frames, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3),
            name_ + "_dynamics_constraint",
            deriv_lib_path,
            ad::DerivativeOrder::FirstOrder, 3*vel_dim_ + tau_dim_ + CONTACT_3DOF*num_contacts_,
            1 + config_dim_ + 2*vel_dim_ + tau_dim_ + CONTACT_3DOF*num_contacts_,
            compile_derivs
        );

        integration_function_ = std::make_unique<ad::CppADInterface>(
            std::bind(&DynamicsConstraint::IntegrationConstraint, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
            name_ + "_integration_constraint",
            deriv_lib_path,
            ad::DerivativeOrder::FirstOrder, 3*vel_dim_, 1 + 2*config_dim_ + vel_dim_,
            compile_derivs
        );

        // Load all the functions
        std::cout << "Looking for casadi libs at " << std::filesystem::current_path() << std::endl;
        casadi_dynamics_function_ = std::make_unique<casadi::Function>(casadi::external("dvkp1", "dynamics_functions.so"));
        casadi_dq_jac_function_ = std::make_unique<casadi::Function>(casadi::external("derivatives_fun_dq", "dynamics_functions.so"));
        casadi_dv_jac_function_ = std::make_unique<casadi::Function>(casadi::external("derivatives_fun_dv", "dynamics_functions.so"));
        casadi_dtau_jac_function_ = std::make_unique<casadi::Function>(casadi::external("derivatives_fun_dtau", "dynamics_functions.so"));
        casadi_dF_jac_function_ = std::make_unique<casadi::Function>(casadi::external("derivatives_fun_dF", "dynamics_functions.so"));


        // Try calling the function
        vectorx_t q = model_.GetRandomConfig();
        vectorx_t dq1 = vectorx_t::Random(model_.GetVelDim());
        // dq1.segment<3>(3).setZero();
        // dq1(3) = 0.1;
        Eigen::Quaternion<double> quat(q.template segment<4>(3));
        // std::cerr << "quat <vec>: " << q.segment<4>(3).transpose() << std::endl;
        // std::cerr << "quat: " << quat << std::endl;
        // std::cerr << "dquat: " << dq1.segment<3>(3).transpose() << std::endl;
        // std::cerr << "exp3: " << pinocchio::quaternion::exp3(dq1.segment<3>(3)) << std::endl;
        // std::cerr << "quat mult: " << quat * pinocchio::quaternion::exp3(dq1.segment<3>(3)) << std::endl;
        vectorx_t v = vectorx_t::Random(model_.GetVelDim());
        vectorx_t dv1 = vectorx_t::Random(model_.GetVelDim());
        vectorx_t v2 = vectorx_t::Random(model_.GetVelDim());
        vectorx_t tau = vectorx_t::Random(model_.GetVelDim());
        vectorx_t dtau = vectorx_t::Random(model_.GetVelDim());
        vectorx_t F = vectorx_t::Random(CONTACT_3DOF*num_contacts_);
        vectorx_t dF = vectorx_t::Random(CONTACT_3DOF*num_contacts_);
        double dt = 0.015;

        matrixx_t Jdq, Jdv, Jdtau, JdF;
        vectorx_t b;

        ComputeDynamicsJacobians(q, v, v2, tau, F, dt, dq1, dv1, dtau, dF, Jdq, Jdv, Jdtau, JdF, b);

        // std::cout << "Jdq:\n" << Jdq << std::endl;
        // std::cout << "Jdv:\n" << Jdv << std::endl;
        // std::cout << "Jdtau:\n" << Jdtau << std::endl;
        // std::cout << "JdF:\n" << JdF << std::endl;

        // std::vector<double> q_vec((size_t)model_.GetConfigDim());
        // Eigen::Map<Eigen::VectorXd>(q_vec.data(), model_.GetConfigDim(), 1) = q;
        //
        // std::vector<double> v_vec((size_t)model_.GetVelDim());
        // Eigen::Map<Eigen::VectorXd>(v_vec.data(), model_.GetVelDim(), 1) = v;
        //
        // std::vector<double> tau_vec((size_t)model_.GetVelDim());
        // Eigen::Map<Eigen::VectorXd>(tau_vec.data(), model_.GetVelDim(), 1) = tau;
        //
        // std::vector<double> F_vec((size_t)(CONTACT_3DOF*num_contacts_));
        // Eigen::Map<Eigen::VectorXd>(F_vec.data(), CONTACT_3DOF*num_contacts_, 1) = F;
        //
        // casadi::DM a_res = (*casadi_dynamics_function_)(casadi::DMVector{q_vec, v_vec, tau_vec, F_vec})[0];
        // vectorx_t a_casadi_vec = Eigen::Map<vectorx_t>(
        //   static_cast<std::vector<double>>(a_res).data(), model_.GetVelDim(), 1);
        //
        // pinocchio::Data data(model_.GetModel());
        //
        // int idx = 0;
        // std::vector<models::ExternalForce<double>> f_ext;
        // for (const auto& frame : contact_frames) {
        //     f_ext.emplace_back(frame, F.segment<CONTACT_3DOF>(idx));
        //     idx += CONTACT_3DOF;
        // }
        // vectorx_t a_pin = models::ForwardDynamics(model_.GetModel(), data, q, v, tau, f_ext);
        //
        // std::cout << "CasADi aba:\n" << "\ta = " << a_casadi_vec.transpose() << std::endl;
        // std::cout << "pinocchio aba:\n" << "\ta = " << a_pin.transpose() << std::endl;

        std::cout << "g1 config dim: " << model_.GetConfigDim() << std::endl;
        std::cout << "g1 vel dim: " << model_.GetVelDim() << std::endl;

        // TODO: To get the "forward dynamics" I will need to invert the jacobian term relating to v2
    }

// void DynamicsConstraint::GetLinDynamics(const vectorx_t &q1_lin, const vectorx_t &q2_lin,
//         const vectorx_t &v1_lin, const vectorx_t& v2_lin, const vectorx_t &tau_lin, const vectorx_t &force_lin,
//         double dt, bool boundary, matrixx_t& A, matrixx_t& B, vectorx_t& b) {
//         // --------- Dynamics --------- //
//         matrixx_t Jdq, Jdv, Jdtau, JdF;
//         vectorx_t dyn_constant;
//         vectorx_t zero_tangent = vectorx_t::Zero(vel_dim_);
//         vectorx_t zero_force = vectorx_t::Zero(CONTACT_3DOF*num_contacts_);
//
//         vectorx_t tau_casadi = vectorx_t::Zero(vel_dim_);
//         tau_casadi.tail(tau_dim_) = tau_lin;
//
//         ComputeDynamicsJacobians(q1_lin, v1_lin, v2_lin, tau_casadi, force_lin, dt,
//             zero_tangent, zero_tangent, zero_tangent, zero_force, Jdq, Jdv, Jdtau, JdF, dyn_constant);
//
//         // --------- Integration --------- //
//         vectorx_t x_zero(integration_function_->GetDomainSize());
//         x_zero.setZero();
//
//         vectorx_t p(integration_function_->GetParameterSize());
//         p << dt, q1_lin, q2_lin, v1_lin;
//
//         // TODO: Think about if the violation that is returned is really the delta that we want
//         matrixx_t int_jac;
//         integration_function_->GetJacobian(x_zero, p, int_jac);
//
//         vectorx_t int_fbar;
//         integration_function_->GetFunctionValue(x_zero, p, int_fbar);
//         // vectorx_t q2_default = pinocchio::integrate(model_.GetModel(), q1_lin, dt*v1_lin);
//         // vectorx_t dq2 = models::qDifference(q2_default, q2_lin);
//
//         if (boundary) {
//             // Try #1: Just use the centroidal dynamics -- must be called from the non-full order node
//             A.setZero();
//             B.setZero();
//             b.setZero();
//
//             matrixx_t dq2_inv = int_jac.middleCols(vel_dim_, vel_dim_).inverse();
//             // std::cout << "dq2_inv:\n" << dq2_inv << std::endl;
//
//             // Integration
//             A.topRows(vel_dim_) << -dq2_inv*int_jac.leftCols(vel_dim_), -dq2_inv*int_jac.middleCols(2*vel_dim_, vel_dim_);
//             B.topRows(vel_dim_).setZero();
//
//             // Dynamics
//             A.bottomRows<FLOATING_VEL>() << Jdq.topRows<FLOATING_VEL>(), Jdv.topRows<FLOATING_VEL>();
//             B.bottomRows<FLOATING_VEL>() << Jdtau.topRows<FLOATING_VEL>(), JdF.topRows<FLOATING_VEL>();
//
//             b << -dq2_inv*int_fbar, dyn_constant.head<FLOATING_VEL>();
//         } else if (full_order_) {
//             A.setZero();
//             B.setZero();
//             b.setZero();
//
//             matrixx_t dq2_inv = int_jac.middleCols(vel_dim_, vel_dim_).inverse();
//             // std::cout << "dq2_inv:\n" << dq2_inv << std::endl;
//
//             // Integration
//             A.topRows(vel_dim_) << -dq2_inv*int_jac.leftCols(vel_dim_), -dq2_inv*int_jac.middleCols(2*vel_dim_, vel_dim_);
//             B.topRows(vel_dim_).setZero();
//
//             // Dynamics
//             A.bottomRows(vel_dim_) << Jdq, Jdv;
//             B.bottomRows(vel_dim_) << Jdtau, JdF;
//
//             // Constant
//             b << -dq2_inv*int_fbar, dyn_constant;
//         } else {
//             A.setZero();
//             B.setZero();
//             b.setZero();
//
//             matrixx_t dq2_inv = int_jac.middleCols(vel_dim_, vel_dim_).inverse();
//             // std::cout << "dq2_inv:\n" << dq2_inv << std::endl;
//
//             // Integration
//             A.topRows(vel_dim_) << -dq2_inv*int_jac.leftCols(vel_dim_), -dq2_inv*int_jac.middleCols(2*vel_dim_, FLOATING_VEL);
//             B.topLeftCorner(vel_dim_, vel_dim_ - FLOATING_VEL) = -dq2_inv*int_jac.rightCols(vel_dim_ - FLOATING_VEL);
//
//             // Dynamics
//             A.bottomRows<FLOATING_VEL>() << Jdq.topRows<FLOATING_VEL>(), Jdv.topLeftCorner<FLOATING_VEL, FLOATING_VEL>();
//             B.bottomRows<FLOATING_VEL>() << Jdv.topRightCorner(FLOATING_VEL, vel_dim_ - FLOATING_VEL), JdF.topRows<FLOATING_VEL>();
//
//             b << -dq2_inv*int_fbar, dyn_constant.head<FLOATING_VEL>();
//         }
//     }

    void DynamicsConstraint::GetLinDynamics(const vectorx_t &q1_lin, const vectorx_t &q2_lin,
        const vectorx_t &v1_lin, const vectorx_t& v2_lin, const vectorx_t &tau_lin, const vectorx_t &force_lin,
        double dt, bool boundary, matrixx_t& A, matrixx_t& B, vectorx_t& b) {
        // --------- Dynamics --------- //
        vectorx_t x_zero = vectorx_t::Zero(dynamics_function_->GetDomainSize());
        vectorx_t p(dynamics_function_->GetParameterSize());
        p << q1_lin, v1_lin, v2_lin, tau_lin, force_lin, dt;

        matrixx_t dyn_jac;
        dynamics_function_->GetJacobian(x_zero, p, dyn_jac);

        vectorx_t fbar;
        dynamics_function_->GetFunctionValue(x_zero, p, fbar);
        // pinocchio::Data data(model_.GetModel()); // TODO: Move this to the constructor
        // vectorx_t tau_for_pin(v1_lin.size());
        // tau_for_pin << vectorx_t::Zero(FLOATING_VEL), tau_lin;
        // std::vector<models::ExternalForce<double>> f_ext;
        // for (int i = 0; i < num_contacts_; i++) {
        //     f_ext.emplace_back(contact_frames_[i], force_lin.segment<3>(3*i));
        // }
        //
        // vectorx_t a_default = models::ForwardDynamics(model_.GetModel(), data, q1_lin, v1_lin, tau_for_pin, f_ext);
        // vectorx_t v2_default = a_default*dt + v1_lin;
        // vectorx_t dv2 = v2_default - v2_lin;

        // --------- Integration --------- //
        x_zero.resize(integration_function_->GetDomainSize());
        x_zero.setZero();

        p.resize(integration_function_->GetParameterSize());
        p << dt, q1_lin, q2_lin, v1_lin;

        // TODO: Think about if the violation that is returned is really the delta that we want
        matrixx_t int_jac;
        integration_function_->GetJacobian(x_zero, p, int_jac);

        vectorx_t int_fbar;
        integration_function_->GetFunctionValue(x_zero, p, int_fbar);
        // vectorx_t q2_default = pinocchio::integrate(model_.GetModel(), q1_lin, dt*v1_lin);
        // vectorx_t dq2 = models::qDifference(q2_default, q2_lin);

        if (boundary) {
            // Try #1: Just use the centroidal dynamics -- must be called from the non-full order node
            A.setZero();
            B.setZero();
            b.setZero();

            // TODO: Check to be sure this is grabbing the block I want
            // std::cerr << "jv2:\n" << dyn_jac.block(0, 2*vel_dim_, FLOATING_VEL, FLOATING_VEL) << std::endl;
            if (dyn_jac.rows() != FLOATING_VEL) {
                throw std::runtime_error("dyn jac wrong number of rows!");
            }
            matrixx_t dv2_inv = dyn_jac.middleCols(2*vel_dim_, FLOATING_VEL).inverse();
            matrixx_t dq2_inv = int_jac.middleCols(vel_dim_, vel_dim_).inverse();
            // std::cout << "dq2_inv:\n" << dq2_inv << std::endl;

            A.topRows(vel_dim_) << -dq2_inv*int_jac.leftCols(vel_dim_), -dq2_inv*int_jac.middleCols(2*vel_dim_, vel_dim_);
            A.bottomRows<FLOATING_VEL>() << -dv2_inv*dyn_jac.leftCols(2*vel_dim_);

            B.bottomRows<FLOATING_VEL>() << dyn_jac.middleCols(3*vel_dim_, tau_dim_),
                dyn_jac.middleCols(3*vel_dim_ + tau_dim_, CONTACT_3DOF*num_contacts_);
            B.bottomRows<FLOATING_VEL>() = -dv2_inv*B.bottomRows<FLOATING_VEL>();

            // b << int_fbar, -dv2_inv*fbar.head<FLOATING_VEL>();
            b << -dq2_inv*int_fbar, -dv2_inv*fbar;
        } else if (full_order_) {
            A.setZero();
            B.setZero();
            b.setZero();

            matrixx_t dv2_inv = dyn_jac.middleCols(2*vel_dim_, vel_dim_).inverse();
            // std::cout << "dv2_inv:\n" << dv2_inv << std::endl;
            matrixx_t dq2_inv = int_jac.middleCols(vel_dim_, vel_dim_).inverse();
            // std::cout << "dq2_inv:\n" << dq2_inv << std::endl;

            A.topRows(vel_dim_) << -dq2_inv*int_jac.leftCols(vel_dim_), -dq2_inv*int_jac.middleCols(2*vel_dim_, vel_dim_);
            A.bottomRows(vel_dim_) = -dv2_inv*dyn_jac.leftCols(2*vel_dim_);

            B.topRows(vel_dim_).setZero();
            // TODO: Put torque part back
            B.bottomRows(vel_dim_) << dyn_jac.middleCols(3*vel_dim_, tau_dim_),
                dyn_jac.middleCols(3*vel_dim_ + tau_dim_, CONTACT_3DOF*num_contacts_);
            // B.block(vel_dim_, tau_dim_, vel_dim_, CONTACT_3DOF*num_contacts_)
            //     = -dv2_inv*dyn_jac.middleCols(3*vel_dim_ + tau_dim_, CONTACT_3DOF*num_contacts_);
            B.bottomRows(vel_dim_) = -dv2_inv*B.bottomRows(vel_dim_);
            // B.middleRows(vel_dim_, FLOATING_VEL).setZero();  // TODO: Remove

            // std::cout << "dv2_inv:\n" << dv2_inv << std::endl;
            // std::cout << "Jtau:\n" << dyn_jac.middleCols(3*vel_dim_, tau_dim_) << std::endl;
            // std::cout << "dv2inv*Jtau:\n" << dv2_inv * dyn_jac.middleCols(3*vel_dim_, tau_dim_) << std::endl;

            // b << int_fbar, -dv2_inv*fbar;
            b << -dq2_inv*int_fbar, -dv2_inv*fbar;
        } else {
            A.setZero();
            B.setZero();
            b.setZero();

            // TODO: Check to be sure this is grabbing the block I want
            // std::cerr << "jv2:\n" << dyn_jac.block(0, 2*vel_dim_, FLOATING_VEL, FLOATING_VEL) << std::endl;
            if (dyn_jac.rows() != FLOATING_VEL) {
                throw std::runtime_error("dyn jac wrong number of rows!");
            }
            matrixx_t dv2_inv = dyn_jac.middleCols(2*vel_dim_, FLOATING_VEL).inverse();
            matrixx_t dq2_inv = int_jac.middleCols(vel_dim_, vel_dim_).inverse();
            // std::cout << "dq2_inv:\n" << dq2_inv << std::endl;

            A.topRows(vel_dim_) << -dq2_inv*int_jac.leftCols(vel_dim_), -dq2_inv*int_jac.middleCols(2*vel_dim_, FLOATING_VEL);
            // A.topRightCorner<FLOATING_VEL, FLOATING_VEL>() =
            //     int_jac.block(0, 2*vel_dim_, FLOATING_VEL, FLOATING_VEL);
            A.bottomRows<FLOATING_VEL>() << -dv2_inv*dyn_jac.leftCols(vel_dim_ + FLOATING_VEL);

            B.topLeftCorner(vel_dim_, vel_dim_ - FLOATING_VEL) = -dq2_inv*int_jac.rightCols(vel_dim_ - FLOATING_VEL);
            // B.block(FLOATING_VEL, 0, vel_dim_ - FLOATING_VEL, vel_dim_ - FLOATING_VEL) =
            //     int_jac.block(FLOATING_VEL, 2*vel_dim_ + FLOATING_VEL, vel_dim_ - FLOATING_VEL, vel_dim_ - FLOATING_VEL);

            B.bottomRows<FLOATING_VEL>() << dyn_jac.middleCols(vel_dim_ + FLOATING_VEL, vel_dim_ - FLOATING_VEL),
                dyn_jac.rightCols(num_contacts_*CONTACT_3DOF);
            B.bottomRows<FLOATING_VEL>() = -dv2_inv*B.bottomRows<FLOATING_VEL>();

            // b << int_fbar, -dv2_inv*fbar.head<FLOATING_VEL>();
            b << -dq2_inv*int_fbar, -dv2_inv*fbar.head<FLOATING_VEL>();
        }
    }

    //  void DynamicsConstraint::GetLinDynamics(const vectorx_t &q1_lin, const vectorx_t &q2_lin,
    //     const vectorx_t &v1_lin, const vectorx_t& v2_lin, const vectorx_t &tau_lin, const vectorx_t &force_lin,
    //     double dt, bool boundary, matrixx_t& A, matrixx_t& B, vectorx_t& b) {
    //     // =========== COMPARISON =========== //
    //     // --------- Dynamics --------- //
    //     vectorx_t x_zero = vectorx_t::Zero(dynamics_function_->GetDomainSize());
    //     vectorx_t p(dynamics_function_->GetParameterSize());
    //     p << q1_lin, v1_lin, v2_lin, tau_lin, force_lin, dt;
    //
    //     matrixx_t dyn_jac;
    //     dynamics_function_->GetJacobian(x_zero, p, dyn_jac);
    //
    //     vectorx_t fbar;
    //     dynamics_function_->GetFunctionValue(x_zero, p, fbar);
    //
    //     // --------- Integration --------- //
    //     x_zero.resize(integration_function_->GetDomainSize());
    //     x_zero.setZero();
    //
    //     p.resize(integration_function_->GetParameterSize());
    //     p << dt, q1_lin, q2_lin, v1_lin;
    //
    //     // TODO: Think about if the violation that is returned is really the delta that we want
    //     matrixx_t int_jac;
    //     integration_function_->GetJacobian(x_zero, p, int_jac);
    //
    //     vectorx_t int_fbar;
    //     integration_function_->GetFunctionValue(x_zero, p, int_fbar);
    //     // vectorx_t q2_default = pinocchio::integrate(model_.GetModel(), q1_lin, dt*v1_lin);
    //     // vectorx_t dq2 = models::qDifference(q2_default, q2_lin);
    //
    //     if (boundary) {
    //         // Try #1: Just use the centroidal dynamics -- must be called from the non-full order node
    //         A.setZero();
    //         B.setZero();
    //         b.setZero();
    //
    //         // TODO: Check to be sure this is grabbing the block I want
    //         // std::cerr << "jv2:\n" << dyn_jac.block(0, 2*vel_dim_, FLOATING_VEL, FLOATING_VEL) << std::endl;
    //         if (dyn_jac.rows() != FLOATING_VEL) {
    //             throw std::runtime_error("dyn jac wrong number of rows!");
    //         }
    //         matrixx_t dv2_inv = dyn_jac.middleCols(2*vel_dim_, FLOATING_VEL).inverse();
    //         matrixx_t dq2_inv = int_jac.middleCols(vel_dim_, vel_dim_).inverse();
    //         // std::cout << "dq2_inv:\n" << dq2_inv << std::endl;
    //
    //         A.topRows(vel_dim_) << -dq2_inv*int_jac.leftCols(vel_dim_), -dq2_inv*int_jac.middleCols(2*vel_dim_, vel_dim_);
    //         A.bottomRows<FLOATING_VEL>() << -dv2_inv*dyn_jac.leftCols(2*vel_dim_);
    //
    //         B.bottomRows<FLOATING_VEL>() << dyn_jac.middleCols(3*vel_dim_, tau_dim_),
    //             dyn_jac.middleCols(3*vel_dim_ + tau_dim_, CONTACT_3DOF*num_contacts_);
    //         B.bottomRows<FLOATING_VEL>() = -dv2_inv*B.bottomRows<FLOATING_VEL>();
    //
    //         // b << int_fbar, -dv2_inv*fbar.head<FLOATING_VEL>();
    //         b << -dq2_inv*int_fbar, -dv2_inv*fbar;
    //     } else if (full_order_) {
    //         A.setZero();
    //         B.setZero();
    //         b.setZero();
    //
    //         matrixx_t dv2_inv = dyn_jac.middleCols(2*vel_dim_, vel_dim_).inverse();
    //         // std::cout << "dv2_inv:\n" << dv2_inv << std::endl;
    //         matrixx_t dq2_inv = int_jac.middleCols(vel_dim_, vel_dim_).inverse();
    //         // std::cout << "dq2_inv:\n" << dq2_inv << std::endl;
    //
    //         A.topRows(vel_dim_) << -dq2_inv*int_jac.leftCols(vel_dim_), -dq2_inv*int_jac.middleCols(2*vel_dim_, vel_dim_);
    //         A.bottomRows(vel_dim_) = -dv2_inv*dyn_jac.leftCols(2*vel_dim_);
    //
    //         B.topRows(vel_dim_).setZero();
    //         // TODO: Put torque part back
    //         B.bottomRows(vel_dim_) << dyn_jac.middleCols(3*vel_dim_, tau_dim_),
    //             dyn_jac.middleCols(3*vel_dim_ + tau_dim_, CONTACT_3DOF*num_contacts_);
    //         // B.block(vel_dim_, tau_dim_, vel_dim_, CONTACT_3DOF*num_contacts_)
    //         //     = -dv2_inv*dyn_jac.middleCols(3*vel_dim_ + tau_dim_, CONTACT_3DOF*num_contacts_);
    //         B.bottomRows(vel_dim_) = -dv2_inv*B.bottomRows(vel_dim_);
    //         // B.middleRows(vel_dim_, FLOATING_VEL).setZero();  // TODO: Remove
    //
    //         // std::cout << "dv2_inv:\n" << dv2_inv << std::endl;
    //         // std::cout << "Jtau:\n" << dyn_jac.middleCols(3*vel_dim_, tau_dim_) << std::endl;
    //         // std::cout << "dv2inv*Jtau:\n" << dv2_inv * dyn_jac.middleCols(3*vel_dim_, tau_dim_) << std::endl;
    //
    //         // b << int_fbar, -dv2_inv*fbar;
    //         b << -dq2_inv*int_fbar, -dv2_inv*fbar;
    //     } else {
    //         A.setZero();
    //         B.setZero();
    //         b.setZero();
    //
    //         // TODO: Check to be sure this is grabbing the block I want
    //         // std::cerr << "jv2:\n" << dyn_jac.block(0, 2*vel_dim_, FLOATING_VEL, FLOATING_VEL) << std::endl;
    //         if (dyn_jac.rows() != FLOATING_VEL) {
    //             throw std::runtime_error("dyn jac wrong number of rows!");
    //         }
    //         matrixx_t dv2_inv = dyn_jac.middleCols(2*vel_dim_, FLOATING_VEL).inverse();
    //         matrixx_t dq2_inv = int_jac.middleCols(vel_dim_, vel_dim_).inverse();
    //         // std::cout << "dq2_inv:\n" << dq2_inv << std::endl;
    //
    //         A.topRows(vel_dim_) << -dq2_inv*int_jac.leftCols(vel_dim_), -dq2_inv*int_jac.middleCols(2*vel_dim_, FLOATING_VEL);
    //         // A.topRightCorner<FLOATING_VEL, FLOATING_VEL>() =
    //         //     int_jac.block(0, 2*vel_dim_, FLOATING_VEL, FLOATING_VEL);
    //         A.bottomRows<FLOATING_VEL>() << -dv2_inv*dyn_jac.leftCols(vel_dim_ + FLOATING_VEL);
    //
    //         B.topLeftCorner(vel_dim_, vel_dim_ - FLOATING_VEL) = -dq2_inv*int_jac.rightCols(vel_dim_ - FLOATING_VEL);
    //         // B.block(FLOATING_VEL, 0, vel_dim_ - FLOATING_VEL, vel_dim_ - FLOATING_VEL) =
    //         //     int_jac.block(FLOATING_VEL, 2*vel_dim_ + FLOATING_VEL, vel_dim_ - FLOATING_VEL, vel_dim_ - FLOATING_VEL);
    //
    //         B.bottomRows<FLOATING_VEL>() << dyn_jac.middleCols(vel_dim_ + FLOATING_VEL, vel_dim_ - FLOATING_VEL),
    //             dyn_jac.rightCols(num_contacts_*CONTACT_3DOF);
    //         B.bottomRows<FLOATING_VEL>() = -dv2_inv*B.bottomRows<FLOATING_VEL>();
    //
    //         // b << int_fbar, -dv2_inv*fbar.head<FLOATING_VEL>();
    //         b << -dq2_inv*int_fbar, -dv2_inv*fbar.head<FLOATING_VEL>();
    //     }
    //
    //     // std::cout << "A ID:\n" << A << std::endl;
    //     std::cout << "B ID:\n" << B << std::endl;
    //     // std::cout << "b ID:" << b.transpose() << std::endl;
    //
    //      // --------- Dynamics --------- //
    //      matrixx_t Jdq, Jdv, Jdtau, JdF;
    //      vectorx_t dyn_constant;
    //      vectorx_t zero_tangent = vectorx_t::Zero(vel_dim_);
    //      vectorx_t zero_force = vectorx_t::Zero(CONTACT_3DOF*num_contacts_);
    //
    //      vectorx_t tau_casadi = vectorx_t::Zero(vel_dim_);
    //      tau_casadi.tail(tau_dim_) = tau_lin;
    //
    //      ComputeDynamicsJacobians(q1_lin, v1_lin, v2_lin, tau_casadi, force_lin, dt,
    //          zero_tangent, zero_tangent, zero_tangent, zero_force, Jdq, Jdv, Jdtau, JdF, dyn_constant);
    //
    //     // --------- Integration --------- //
    //     x_zero.resize(integration_function_->GetDomainSize());
    //     x_zero.setZero();
    //
    //     p.resize(integration_function_->GetParameterSize());
    //     p << dt, q1_lin, q2_lin, v1_lin;
    //
    //      // TODO: Think about if the violation that is returned is really the delta that we want
    //     int_jac.setZero();
    //     integration_function_->GetJacobian(x_zero, p, int_jac);
    //
    //     int_fbar.setZero();
    //     integration_function_->GetFunctionValue(x_zero, p, int_fbar);
    //
    //      if (boundary) {
    //          // Try #1: Just use the centroidal dynamics -- must be called from the non-full order node
    //          A.setZero();
    //          B.setZero();
    //          b.setZero();
    //
    //          matrixx_t dq2_inv = int_jac.middleCols(vel_dim_, vel_dim_).inverse();
    //          // std::cout << "dq2_inv:\n" << dq2_inv << std::endl;
    //
    //          // Integration
    //          A.topRows(vel_dim_) << -dq2_inv*int_jac.leftCols(vel_dim_), -dq2_inv*int_jac.middleCols(2*vel_dim_, vel_dim_);
    //          B.topRows(vel_dim_).setZero();
    //
    //          // Dynamics
    //          A.bottomRows<FLOATING_VEL>() << Jdq.topRows<FLOATING_VEL>(), Jdv.topRows<FLOATING_VEL>();
    //          B.bottomRows<FLOATING_VEL>() << Jdtau.topRows<FLOATING_VEL>(), JdF.topRows<FLOATING_VEL>();
    //
    //          b << -dq2_inv*int_fbar, dyn_constant.head<FLOATING_VEL>();
    //      } else if (full_order_) {
    //          A.setZero();
    //          B.setZero();
    //          b.setZero();
    //
    //          matrixx_t dq2_inv = int_jac.middleCols(vel_dim_, vel_dim_).inverse();
    //          // std::cout << "dq2_inv:\n" << dq2_inv << std::endl;
    //
    //          // Integration
    //          A.topRows(vel_dim_) << -dq2_inv*int_jac.leftCols(vel_dim_), -dq2_inv*int_jac.middleCols(2*vel_dim_, vel_dim_);
    //          B.topRows(vel_dim_).setZero();
    //
    //          // Dynamics
    //          A.bottomRows(vel_dim_) << Jdq, Jdv;
    //          B.bottomRows(vel_dim_) << Jdtau, JdF;
    //
    //          // Constant
    //          b << -dq2_inv*int_fbar, dyn_constant;
    //      } else {
    //          A.setZero();
    //          B.setZero();
    //          b.setZero();
    //
    //          matrixx_t dq2_inv = int_jac.middleCols(vel_dim_, vel_dim_).inverse();
    //          // std::cout << "dq2_inv:\n" << dq2_inv << std::endl;
    //
    //          // Integration
    //          A.topRows(vel_dim_) << -dq2_inv*int_jac.leftCols(vel_dim_), -dq2_inv*int_jac.middleCols(2*vel_dim_, FLOATING_VEL);
    //          B.topLeftCorner(vel_dim_, vel_dim_ - FLOATING_VEL) = -dq2_inv*int_jac.rightCols(vel_dim_ - FLOATING_VEL);
    //
    //          // Dynamics
    //          A.bottomRows<FLOATING_VEL>() << Jdq.topRows<FLOATING_VEL>(), Jdv.topLeftCorner<FLOATING_VEL, FLOATING_VEL>();
    //          B.bottomRows<FLOATING_VEL>() << Jdv.topRightCorner(FLOATING_VEL, vel_dim_ - FLOATING_VEL), JdF.topRows<FLOATING_VEL>();
    //
    //          b << -dq2_inv*int_fbar, dyn_constant.head<FLOATING_VEL>();
    //      }
    //
    //     // std::cout << "A FD:\n" << A << std::endl;
    //     std::cout << "B FD:\n" << B << std::endl;
    //     // std::cout << "b FD:" << b.transpose() << std::endl;
    // }

    // // DEBUG -- Finite Difference
    // void DynamicsConstraint::GetLinDynamics(const vectorx_t &q1_lin, const vectorx_t &q2_lin, const vectorx_t &v1_lin,
    //     const vectorx_t& v2_lin, const vectorx_t &tau_lin, const vectorx_t &force_lin, double dt,
    //      matrixx_t& A, matrixx_t& B, vectorx_t& b) {
    //     const double FD_DELTA = 1e-8;
    //     pinocchio::Model model = model_.GetModel();
    //     pinocchio::Data data(model);
    //
    //     std::vector<models::ExternalForce<double>> f_ext;
    //     for (int i = 0; i < num_contacts_; i++) {
    //         f_ext.emplace_back(contact_frames_[i], force_lin.segment<3>(3*i));
    //     }
    //
    //     vectorx_t tau_for_pin(vel_dim_);
    //     tau_for_pin << vectorx_t::Zero(FLOATING_VEL), tau_lin;
    //
    //     vectorx_t a_default = models::ForwardDynamics(model, data, q1_lin, v1_lin, tau_for_pin, f_ext);
    //     vectorx_t v2_default = a_default*dt + v1_lin;
    //     vectorx_t dv2 = v2_default - v2_lin;
    //
    //     vectorx_t q2_default = pinocchio::integrate(model, q1_lin, dt*v1_lin);
    //     // std::cout << "q2_default: " << q2_default.transpose() << std::endl;
    //     // std::cout << "q2_lin: " << q2_lin.transpose() << std::endl;
    //     vectorx_t dq2 = models::qDifference(q2_default, q2_lin);
    //     // std::cout << "dq2: " << dq2.transpose() << std::endl;
    //
    //     // ----- Compute FD for q ----- //
    //     matrixx_t Jq = matrixx_t::Zero(model_.GetVelDim(), model_.GetVelDim());
    //     matrixx_t Jintq = matrixx_t::Zero(model_.GetVelDim(), model_.GetVelDim());
    //
    //     for (int col = 0; col < Jq.cols(); col++) {
    //         vectorx_t dq = vectorx_t::Zero(vel_dim_);
    //         dq(col) += FD_DELTA;
    //
    //         vectorx_t q = models::ConvertdqToq(dq, q1_lin);
    //         vectorx_t a = models::ForwardDynamics(model, data, q, v1_lin, tau_for_pin, f_ext);
    //         vectorx_t v2 = dt*a + v1_lin;
    //         vectorx_t dv2_new = v2 - v2_lin;
    //
    //         vectorx_t q2 = pinocchio::integrate(model, q, dt*v1_lin);
    //         vectorx_t dq2_new = models::qDifference(q2, q2_lin);
    //
    //         Jq.col(col) = (dv2_new - dv2)/FD_DELTA;
    //         Jintq.col(col) = (dq2_new - dq2)/FD_DELTA;
    //     }
    //
    //     // ----- Compute FD for v ----- //
    //     matrixx_t Jv = matrixx_t::Zero(model_.GetVelDim(), model_.GetVelDim());
    //     matrixx_t Jintv = matrixx_t::Zero(model_.GetVelDim(), model_.GetVelDim());
    //
    //     for (int col = 0; col < Jv.cols(); col++) {
    //         vectorx_t dv = vectorx_t::Zero(vel_dim_);
    //         dv(col) += FD_DELTA;
    //
    //         vectorx_t v = v1_lin + dv;
    //         vectorx_t a = models::ForwardDynamics(model, data, q1_lin, v, tau_for_pin, f_ext);
    //         vectorx_t v2 = dt*a + v;
    //         vectorx_t dv2_new = v2 - v2_lin;
    //
    //         vectorx_t q2 = pinocchio::integrate(model, q1_lin, dt*v);
    //         vectorx_t dq2_new = models::qDifference(q2, q2_lin);
    //
    //         Jv.col(col) = (dv2_new - dv2)/FD_DELTA;
    //         Jintv.col(col) = (dq2_new - dq2)/FD_DELTA;
    //     }
    //
    //     // ----- Compute FD for tau ----- //
    //     matrixx_t Jtau = matrixx_t::Zero(model_.GetVelDim(), tau_dim_);
    //
    //     for (int col = 0; col < Jtau.cols(); col++) {
    //         vectorx_t dtau = vectorx_t::Zero(tau_dim_);
    //         dtau(col) += FD_DELTA;
    //
    //         vectorx_t tau = tau_lin + dtau;
    //
    //         vectorx_t tau_app(vel_dim_);
    //         tau_app << vectorx_t::Zero(FLOATING_VEL), tau;
    //
    //         vectorx_t a = models::ForwardDynamics(model, data, q1_lin, v1_lin, tau_app, f_ext);
    //         vectorx_t v2 = dt*a + v1_lin;
    //         vectorx_t dv2_new = v2 - v2_lin;
    //
    //         Jtau.col(col) = (dv2_new - dv2)/FD_DELTA;
    //     }
    //
    //     // ----- Compute FD for F ----- //
    //     matrixx_t JF = matrixx_t::Zero(model_.GetVelDim(), CONTACT_3DOF*num_contacts_);
    //
    //     for (int col = 0; col < JF.cols(); col++) {
    //         vectorx_t dF = vectorx_t::Zero(CONTACT_3DOF*num_contacts_);
    //         dF(col) += FD_DELTA;
    //
    //
    //         vectorx_t F = force_lin + dF;
    //
    //         std::vector<models::ExternalForce<double>> f_ext_2;
    //         for (int i = 0; i < num_contacts_; i++) {
    //             f_ext_2.emplace_back(contact_frames_[i], F.segment<3>(3*i));
    //         }
    //
    //         vectorx_t a = models::ForwardDynamics(model, data, q1_lin, v1_lin, tau_for_pin, f_ext_2);
    //         vectorx_t v2 = dt*a + v1_lin;
    //         vectorx_t dv2_new = v2 - v2_lin;
    //
    //         JF.col(col) = (dv2_new - dv2)/FD_DELTA;
    //     }
    //
    //     // ----- Construct mats ----- //
    //     if (full_order_) {
    //         A.setZero();
    //         B.setZero();
    //         b.setZero();
    //
    //         A.topLeftCorner(vel_dim_, vel_dim_) = Jintq;
    //         A.topRightCorner(vel_dim_, vel_dim_) = Jintv;
    //
    //         A.bottomLeftCorner(vel_dim_, vel_dim_) = Jq;
    //         A.bottomRightCorner(vel_dim_, vel_dim_) = Jv;
    //
    //         B.bottomLeftCorner(vel_dim_, tau_dim_) = Jtau;
    //         B.bottomRightCorner(vel_dim_, CONTACT_3DOF*num_contacts_) = JF;
    //
    //         b.head(vel_dim_) = dq2;
    //         b.tail(vel_dim_) = dv2;
    //     } else {
    //         A.setZero();
    //         B.setZero();
    //         b.setZero();
    //
    //         A.topLeftCorner(vel_dim_, vel_dim_) = Jintq;
    //         A.topRightCorner(vel_dim_, FLOATING_VEL) = Jintv.leftCols<FLOATING_VEL>();
    //
    //         A.bottomLeftCorner(FLOATING_VEL, vel_dim_) = Jq.topRows<FLOATING_VEL>();
    //         A.bottomRightCorner(FLOATING_VEL, FLOATING_VEL) = Jv.topLeftCorner<FLOATING_VEL, FLOATING_VEL>();
    //
    //         B.topLeftCorner(vel_dim_, tau_dim_) = Jintv.rightCols(tau_dim_);
    //
    //         B.bottomLeftCorner(FLOATING_VEL, tau_dim_) = Jv.topRightCorner(FLOATING_VEL, tau_dim_);
    //         B.bottomRightCorner(FLOATING_VEL, CONTACT_3DOF*num_contacts_) = JF.topRows<FLOATING_VEL>();
    //
    //         b.head(vel_dim_) = dq2;
    //         b.tail<FLOATING_VEL>() = dv2.head<FLOATING_VEL>();
    //     }
    // }


    void DynamicsConstraint::InverseDynamics(const std::vector<std::string> &frames,
        const ad::ad_vector_t &dqk_dvk_dvkp1_dtauk_dfk, const ad::ad_vector_t &qk_vk_vkp1_tauk_fk_dt,
        ad::ad_vector_t &violation) {
        // Decision variables
        const ad::ad_vector_t& dqk = dqk_dvk_dvkp1_dtauk_dfk.head(vel_dim_);
        const ad::ad_vector_t& dvk = dqk_dvk_dvkp1_dtauk_dfk.segment(vel_dim_, vel_dim_);
        const ad::ad_vector_t& dvkp1 = dqk_dvk_dvkp1_dtauk_dfk.segment(2*vel_dim_, vel_dim_);
        ad::ad_vector_t dtauk(vel_dim_);
        dtauk << ad::ad_vector_t::Zero(FLOATING_VEL), dqk_dvk_dvkp1_dtauk_dfk.segment(3*vel_dim_, tau_dim_);
        const ad::ad_vector_t& dfk = dqk_dvk_dvkp1_dtauk_dfk.segment(3*vel_dim_ + tau_dim_, CONTACT_3DOF*num_contacts_);

        // Reference trajectory
        const ad::ad_vector_t& qk = qk_vk_vkp1_tauk_fk_dt.head(config_dim_);
        const ad::ad_vector_t& vk = qk_vk_vkp1_tauk_fk_dt.segment(config_dim_, vel_dim_);
        const ad::ad_vector_t& vkp1 = qk_vk_vkp1_tauk_fk_dt.segment(config_dim_ + vel_dim_, vel_dim_);
        ad::ad_vector_t tauk(vel_dim_);
        tauk << ad::ad_vector_t::Zero(FLOATING_VEL), qk_vk_vkp1_tauk_fk_dt.segment(config_dim_ + 2*vel_dim_, tau_dim_);
        const ad::ad_vector_t& fk = qk_vk_vkp1_tauk_fk_dt.segment(config_dim_ + 2*vel_dim_ + tau_dim_, CONTACT_3DOF*num_contacts_);
        const ad::adcg_t& dt = qk_vk_vkp1_tauk_fk_dt(config_dim_ + 2*vel_dim_ + tau_dim_ + CONTACT_3DOF*num_contacts_);

        // Current values
        const ad::ad_vector_t qk_curr = models::ConvertdqToq(dqk, qk);
        const ad::ad_vector_t vk_curr = dvk + vk;
        const ad::ad_vector_t vkp1_curr = dvkp1 + vkp1;
        const ad::ad_vector_t tauk_curr = dtauk + tauk;
        const ad::ad_vector_t fk_curr = dfk + fk;

        // Intermediate values
        const ad::ad_vector_t a = (vkp1_curr - vk_curr)/dt; // TODO: Do I need to account for the different local frames somehow?
        std::vector<models::ExternalForce<ad::adcg_t>> f_ext;

        int idx = 0;
        for (const auto& frame : frames) {
            f_ext.emplace_back(frame, fk_curr.segment<CONTACT_3DOF>(idx));
            idx += CONTACT_3DOF;
        }

        // Compute error
        const ad::ad_vector_t tau_id = models::InverseDynamics(model_.GetADPinModel(), *model_.GetADPinData(),
            qk_curr, vk_curr, a, f_ext);

        violation = tau_id - tauk_curr;

        if (!full_order_) {
            violation.conservativeResize(FLOATING_VEL, Eigen::NoChange);
        }
    }

    std::pair<matrixx_t, matrixx_t> DynamicsConstraint::GetBoundaryDynamics() {
        matrixx_t B = matrixx_t::Zero(vel_dim_ + FLOATING_VEL, tau_dim_ + CONTACT_3DOF*num_contacts_);
        matrixx_t A = matrixx_t::Zero(vel_dim_ + FLOATING_VEL, 2*vel_dim_);
        A.topLeftCorner(vel_dim_, vel_dim_) = matrixx_t::Identity(vel_dim_, vel_dim_);

        A.block(vel_dim_, vel_dim_, FLOATING_VEL, FLOATING_VEL)
            = matrixx_t::Identity(FLOATING_VEL, FLOATING_VEL);

        return {A, B};
    }


    int DynamicsConstraint::GetNumConstraints() const {
        return dynamics_function_->GetRangeSize() + integration_function_->GetRangeSize();
    }

    // bool DynamicsConstraint::IsInNodeRange(int node) const {
    //     return node >= first_node_ && node < last_node_;
    // }


     void DynamicsConstraint::IntegrationConstraint(const ad::ad_vector_t& dqk_dqkp1_dvk,
         const ad::ad_vector_t& dt_qkbar_qkp1bar_vk, ad::ad_vector_t& violation) {
        // From the reference trajectory
        const ad::adcg_t& dt = dt_qkbar_qkp1bar_vk(0);
        const ad::ad_vector_t& qkbar = dt_qkbar_qkp1bar_vk.segment(1, config_dim_);
        const ad::ad_vector_t& qkp1bar = dt_qkbar_qkp1bar_vk.segment(1 + config_dim_, config_dim_);
        const ad::ad_vector_t& vkbar = dt_qkbar_qkp1bar_vk.segment(1 + 2*config_dim_, vel_dim_);

        // Changes from decision variables
        const ad::ad_vector_t& dqk = dqk_dqkp1_dvk.head(vel_dim_);
        const ad::ad_vector_t& dqkp1 = dqk_dqkp1_dvk.segment(vel_dim_, vel_dim_);
        const ad::ad_vector_t& dvk = dqk_dqkp1_dvk.segment(2*vel_dim_, vel_dim_);

        // Get the current configuration
        const ad::ad_vector_t qk = torc::models::ConvertdqToq(dqk, qkbar);
        const ad::ad_vector_t qkp1 = torc::models::ConvertdqToq(dqkp1, qkp1bar);

        // Velocity
        const ad::ad_vector_t vk = vkbar + dvk;

        const ad::ad_vector_t v = dt*vk;

        const ad::ad_vector_t qkp1_new = pinocchio::integrate(model_.GetADPinModel(), qk, v);

        // Floating base position differences
        violation.resize(vel_dim_);
        violation.head<POS_VARS>() = qkp1_new.head<POS_VARS>() - qkp1.head<POS_VARS>();

        // Quaternion difference in the tangent space
        Eigen::Quaternion<ad::adcg_t> quat_kp1(qkp1.segment<QUAT_VARS>(POS_VARS));
        Eigen::Quaternion<ad::adcg_t> quat_kp1_new(qkp1_new.segment<QUAT_VARS>(POS_VARS));

        // Eigen's inverse has an if statement, so we can't use it in codegen
        quat_kp1 = Eigen::Quaternion<torc::ad::adcg_t>(quat_kp1.conjugate().coeffs() / quat_kp1.squaredNorm());   // Assumes norm > 0
        violation.segment<3>(POS_VARS) = pinocchio::quaternion::log3(quat_kp1 * quat_kp1_new);

        // Joint differences
        violation.tail(vel_dim_ - FLOATING_VEL) = qkp1_new.tail(config_dim_ - FLOATING_BASE) - qkp1.tail(config_dim_ - FLOATING_BASE);
    }

        // void DynamicsConstraint::ForwardDynamics(const std::vector<std::string> &frames,
        // const ad::ad_vector_t &dqk_dvk_dtauk_dfk, const ad::ad_vector_t &qk_vk_tauk_fk_dt,
        // ad::ad_vector_t &dxdt) {
        // // Decision variables
        // const ad::ad_vector_t& dqk = dqk_dvk_dtauk_dfk.head(vel_dim_);
        // const ad::ad_vector_t& dvk = dqk_dvk_dtauk_dfk.segment(vel_dim_, vel_dim_);
        // ad::ad_vector_t dtauk(vel_dim_);
        // dtauk << ad::ad_vector_t::Zero(FLOATING_VEL), dqk_dvk_dtauk_dfk.segment(2*vel_dim_, tau_dim_);
        // const ad::ad_vector_t& dfk = dqk_dvk_dtauk_dfk.segment(2*vel_dim_ + tau_dim_, CONTACT_3DOF*num_contacts_);
        //
        // // Reference trajectory
        // const ad::ad_vector_t& qk = qk_vk_tauk_fk_dt.head(config_dim_);
        // const ad::ad_vector_t& vk = qk_vk_tauk_fk_dt.segment(config_dim_, vel_dim_);
        // ad::ad_vector_t tauk(vel_dim_);
        // tauk << ad::ad_vector_t::Zero(FLOATING_VEL), qk_vk_tauk_fk_dt.segment(config_dim_ + vel_dim_, tau_dim_);
        // const ad::ad_vector_t& fk = qk_vk_tauk_fk_dt.segment(config_dim_ + vel_dim_ + tau_dim_, CONTACT_3DOF*num_contacts_);
        // const ad::adcg_t& dt = qk_vk_tauk_fk_dt(config_dim_ + vel_dim_ + tau_dim_ + CONTACT_3DOF*num_contacts_);
        //
        // // Current values
        // const ad::ad_vector_t qk_curr = models::ConvertdqToq(dqk, qk);
        // const ad::ad_vector_t vk_curr = dvk + vk;
        // const ad::ad_vector_t tauk_curr = dtauk + tauk;
        // const ad::ad_vector_t fk_curr = dfk + fk;
        //
        // // Intermediate values
        // std::vector<models::ExternalForce<ad::adcg_t>> f_ext;
        //
        // int idx = 0;
        // for (const auto& frame : frames) {
        //     f_ext.emplace_back(frame, fk_curr.segment<CONTACT_3DOF>(idx));
        //     idx += CONTACT_3DOF;
        // }
        //
        // const ad::ad_vector_t a = models::ForwardWithCrba(model_->GetADPinModel(), *model_->GetADPinData(),
        //     qk_curr, vk_curr, tauk_curr, f_ext);
        // const ad::ad_vector_t a = models::ForwardDynamics(model_->GetADPinModel(), *model_->GetADPinData(),
        //     qk_curr, vk_curr, tauk_curr, f_ext);


        // TODO: Or:
        // ad::ad_vector_t vkp1 = vk_curr + a*dt;
        // TODO: Or:
        // ad::ad_vector_t dvk_out = a*dt;
        //
        // // Integration
        // const ad::ad_vector_t qkp1_new = pinocchio::integrate(model_->GetADPinModel(), qk, dt*vk_curr);
        //
        // // Floating base position differences
        // ad::ad_vector_t dqk_out;
        // dqk_out.resize(vk_curr.size());
        // dqk_out.head<POS_VARS>() = qkp1_new.head<POS_VARS>() - qk_curr.head<POS_VARS>();
        //
        // // Quaternion difference in the tangent space
        // Eigen::Quaternion<ad::adcg_t> quat_kp1(qk_curr.segment<QUAT_VARS>(POS_VARS));
        // Eigen::Quaternion<ad::adcg_t> quat_kp1_new(qkp1_new.segment<QUAT_VARS>(POS_VARS));
        //
        // // Eigen's inverse has an if statement, so we can't use it in codegen
        // quat_kp1 = Eigen::Quaternion<torc::ad::adcg_t>(quat_kp1.conjugate().coeffs() / quat_kp1.squaredNorm());   // Assumes norm > 0
        // dqk_out.segment<3>(POS_VARS) = pinocchio::quaternion::log3(quat_kp1 * quat_kp1_new);
        //
        // // Joint differences
        // dqk_out.tail(vk_curr.size() - FLOATING_VEL) = qkp1_new.tail(qk.size() - FLOATING_BASE) - qk_curr.tail(qk.size() - FLOATING_BASE);
        //
        // dxdt.resize(dqk_out.size() + dvk_out.size());
        // dxdt << dvk_out, dqk_out;
    // }

    // std::pair<vectorx_t, vectorx_t> DynamicsConstraint::GetViolation(const vectorx_t &q1_lin, const vectorx_t &q2_lin,
    //     const vectorx_t &v1_lin, const vectorx_t &v2_lin, const vectorx_t &tau_lin,
    //     const vectorx_t &force_lin, double dt, const vectorx_t& dq1, const vectorx_t& dq2,
    //     const vectorx_t& dv1, const vectorx_t& dv2, const vectorx_t& dtau, const vectorx_t& dforce) {
    //     vectorx_t x(dynamics_function_->GetDomainSize());
    //     x << dq1, dv1, dv2, dtau, dforce;
    //
    //     vectorx_t p(dynamics_function_->GetParameterSize());
    //
    //     vectorx_t dyn_violation(dynamics_function_->GetRangeSize());
    //
    //     p << q1_lin, v1_lin, v2_lin, tau_lin, force_lin, dt;
    //     dynamics_function_->GetFunctionValue(x, p, dyn_violation);
    //
    //     vectorx_t int_violation(integration_function_->GetRangeSize());
    //     x.resize(integration_function_->GetDomainSize());
    //     x << dq1, dq2, dv1;
    //     p.resize(integration_function_->GetParameterSize());
    //     p << dt, q1_lin, q2_lin, v1_lin;
    //     integration_function_->GetFunctionValue(x, p, int_violation);
    //
    //     return {int_violation, dyn_violation};
    // }

    std::pair<vectorx_t, vectorx_t> DynamicsConstraint::GetViolation(const vectorx_t &q1_lin, const vectorx_t &q2_lin,
    const vectorx_t &v1_lin, const vectorx_t &v2_lin, const vectorx_t &tau_lin,
    const vectorx_t &force_lin, double dt, const vectorx_t& dq1, const vectorx_t& dq2,
    const vectorx_t& dv1, const vectorx_t& dv2, const vectorx_t& dtau, const vectorx_t& dforce) {
        vectorx_t tau_casadi = vectorx_t::Zero(vel_dim_);
        tau_casadi.tail(tau_dim_) = tau_lin;

        vectorx_t dtau_casadi = vectorx_t::Zero(vel_dim_);
        dtau_casadi.tail(tau_dim_) = dtau;

        // std::vector<double> dq1_vec((size_t)model_.GetVelDim());
        // Eigen::Map<Eigen::VectorXd>(dq1_vec.data(), model_.GetVelDim()) = dq1;
        //
        // std::vector<double> dv1_vec((size_t)model_.GetVelDim());
        // Eigen::Map<Eigen::VectorXd>(dv1_vec.data(), model_.GetVelDim(), 1) = dv1;
        //
        // std::vector<double> dtau1_vec((size_t)model_.GetVelDim());
        // Eigen::Map<Eigen::VectorXd>(dtau1_vec.data(), model_.GetVelDim(), 1) = dtau_casadi;
        //
        // std::vector<double> dF1_vec((size_t)(CONTACT_3DOF*num_contacts_));
        // Eigen::Map<Eigen::VectorXd>(dF1_vec.data(), CONTACT_3DOF*num_contacts_, 1) = dforce;
        //
        //
        // std::vector<double> q1_vec((size_t)model_.GetConfigDim());
        // Eigen::Map<Eigen::VectorXd>(q1_vec.data(), model_.GetConfigDim(), 1) = q1_lin;
        //
        // std::vector<double> v1_vec((size_t)model_.GetVelDim());
        // Eigen::Map<Eigen::VectorXd>(v1_vec.data(), model_.GetVelDim(), 1) = v1_lin;
        //
        // std::vector<double> v2_vec((size_t)model_.GetVelDim());
        // Eigen::Map<Eigen::VectorXd>(v2_vec.data(), model_.GetVelDim(), 1) = v2_lin;
        //
        // std::vector<double> tau1_vec((size_t)model_.GetVelDim());
        // Eigen::Map<Eigen::VectorXd>(tau1_vec.data(), model_.GetVelDim(), 1) = tau_casadi;
        //
        // std::vector<double> F1_vec((size_t)(CONTACT_3DOF*num_contacts_));
        // Eigen::Map<Eigen::VectorXd>(F1_vec.data(), CONTACT_3DOF*num_contacts_, 1) = force_lin;
        //
        //
        // // TODO: Should verify this against the C++
        // casadi::DM dvkp1_res = (*casadi_dynamics_function_)(casadi::DMVector{dq1_vec, dv1_vec, dtau1_vec, dF1_vec,
        //     q1_vec, v1_vec, v2_vec, tau1_vec, F1_vec, dt})[0];
        // vectorx_t dvkp1 = Eigen::Map<vectorx_t>(
        //   static_cast<std::vector<double>>(dvkp1_res).data(), model_.GetVelDim(), 1);


        // Check against C++
        vectorx_t q_eval = models::ConvertdqToq(dq1, q1_lin);
        vectorx_t v_eval = (v1_lin + dv1);
        vectorx_t tau_eval = (tau_casadi + dtau_casadi);
        vectorx_t force_eval = force_lin + dforce;
        pinocchio::Data data(model_.GetModel());

        int idx = 0;
        std::vector<models::ExternalForce<double>> f_ext;
        for (const auto& frame : contact_frames_) {
            f_ext.emplace_back(frame, force_eval.segment<CONTACT_3DOF>(idx));
            idx += CONTACT_3DOF;
        }


        vectorx_t a_pin = models::ForwardDynamics(model_.GetModel(), data, q_eval, v_eval, tau_eval, f_ext);
        vectorx_t cpp_dv2 = dt*a_pin + v_eval - v2_lin;

        vectorx_t xid(dynamics_function_->GetDomainSize());
        xid << dq1, dv1, dv2, dtau, dforce;

        vectorx_t pid(dynamics_function_->GetParameterSize());

        vectorx_t id_violation(dynamics_function_->GetRangeSize());

        pid << q1_lin, v1_lin, v2_lin, tau_lin, force_lin, dt;
        dynamics_function_->GetFunctionValue(xid, pid, id_violation);

        vectorx_t dyn_vio = dv2 - cpp_dv2;

        if (!full_order_) {
            dyn_vio.conservativeResize(FLOATING_VEL, Eigen::NoChange);
        }

        // std::cout << "FD vio: " << dyn_vio.transpose() << std::endl;
        // std::cout << "ID vio: " << id_violation.transpose() << std::endl;

        // std::cout << "v: " << v_eval.transpose() << std::endl;
        // std::cout << "F: " << force_eval.transpose() << std::endl;
        // std::cout << "tau: " << tau_eval.transpose() << std::endl;
        // std::cout << "pin a: " << a_pin.transpose() << std::endl;
        // std::cout << "Casadi dv2: " << dvkp1.transpose() << std::endl;
        // std::cout << "C++ dv2: " << cpp_dv2.transpose() << std::endl;
        // std::cout << "solution dv2: " << dv2.transpose() << std::endl;
        // std::cout << "id vio: |" << id_violation.squaredNorm() << "| " << id_violation.transpose() << std::endl;

        vectorx_t int_violation(integration_function_->GetRangeSize());
        vectorx_t x(integration_function_->GetDomainSize());
        x << dq1, dq2, dv1;
        vectorx_t p(integration_function_->GetParameterSize());
        p << dt, q1_lin, q2_lin, v1_lin;
        integration_function_->GetFunctionValue(x, p, int_violation);

        return {int_violation, dyn_vio};
    }

    void DynamicsConstraint::ComputeDynamicsJacobians(const vectorx_t &q1_lin, const vectorx_t &v1_lin,
        const vectorx_t &v2_lin, const vectorx_t &tau_lin, const vectorx_t &force_lin, double dt,
        const vectorx_t &dq1, const vectorx_t &dv1, const vectorx_t &dtau,
        const vectorx_t &dforce, matrixx_t& Jdq, matrixx_t& Jdv, matrixx_t& Jdtau, matrixx_t& JdF, vectorx_t& b) {

        std::vector<double> dq1_vec((size_t)model_.GetVelDim());
        Eigen::Map<Eigen::VectorXd>(dq1_vec.data(), model_.GetVelDim()) = dq1;

        std::vector<double> dv1_vec((size_t)model_.GetVelDim());
        Eigen::Map<Eigen::VectorXd>(dv1_vec.data(), model_.GetVelDim(), 1) = dv1;

        std::vector<double> dtau1_vec((size_t)model_.GetVelDim());
        Eigen::Map<Eigen::VectorXd>(dtau1_vec.data(), model_.GetVelDim(), 1) = dtau;

        std::vector<double> dF1_vec((size_t)(CONTACT_3DOF*num_contacts_));
        Eigen::Map<Eigen::VectorXd>(dF1_vec.data(), CONTACT_3DOF*num_contacts_, 1) = dforce;


        std::vector<double> q1_vec((size_t)model_.GetConfigDim());
        Eigen::Map<Eigen::VectorXd>(q1_vec.data(), model_.GetConfigDim(), 1) = q1_lin;

        std::vector<double> v1_vec((size_t)model_.GetVelDim());
        Eigen::Map<Eigen::VectorXd>(v1_vec.data(), model_.GetVelDim(), 1) = v1_lin;

        std::vector<double> v2_vec((size_t)model_.GetVelDim());
        Eigen::Map<Eigen::VectorXd>(v2_vec.data(), model_.GetVelDim(), 1) = v2_lin;

        std::vector<double> tau1_vec((size_t)model_.GetVelDim());
        Eigen::Map<Eigen::VectorXd>(tau1_vec.data(), model_.GetVelDim(), 1) = tau_lin;

        std::vector<double> F1_vec((size_t)(CONTACT_3DOF*num_contacts_));
        Eigen::Map<Eigen::VectorXd>(F1_vec.data(), CONTACT_3DOF*num_contacts_, 1) = force_lin;


        // TODO: Should verify this against the C++
        casadi::DM dvkp1_res = (*casadi_dynamics_function_)(casadi::DMVector{dq1_vec, dv1_vec, dtau1_vec, dF1_vec,
            q1_vec, v1_vec, v2_vec, tau1_vec, F1_vec, dt})[0];
        b = Eigen::Map<vectorx_t>(
          static_cast<std::vector<double>>(dvkp1_res).data(), model_.GetVelDim(), 1);

        casadi::DM Jdq_res = (*casadi_dq_jac_function_)(casadi::DMVector{dq1_vec, dv1_vec, dtau1_vec, dF1_vec,
            q1_vec, v1_vec, v2_vec, tau1_vec, F1_vec, dt})[0];
        Jdq = Eigen::Map<matrixx_t>(static_cast<std::vector<double>>(Jdq_res).data(), vel_dim_, vel_dim_);

        casadi::DM Jdv_res = (*casadi_dv_jac_function_)(casadi::DMVector{dq1_vec, dv1_vec, dtau1_vec, dF1_vec,
            q1_vec, v1_vec, v2_vec, tau1_vec, F1_vec, dt})[0];
        Jdv = Eigen::Map<matrixx_t>(static_cast<std::vector<double>>(Jdv_res).data(), vel_dim_, vel_dim_);

        casadi::DM Jdtau_res = (*casadi_dtau_jac_function_)(casadi::DMVector{dq1_vec, dv1_vec, dtau1_vec, dF1_vec,
            q1_vec, v1_vec, v2_vec, tau1_vec, F1_vec, dt})[0];
        matrixx_t Jdtau_temp = Eigen::Map<matrixx_t>(static_cast<std::vector<double>>(Jdtau_res).data(), vel_dim_, tau_dim_);
        Jdtau = Jdtau_temp.rightCols(vel_dim_ - FLOATING_VEL);

        casadi::DM JdF_res = (*casadi_dF_jac_function_)(casadi::DMVector{dq1_vec, dv1_vec, dtau1_vec, dF1_vec,
            q1_vec, v1_vec, v2_vec, tau1_vec, F1_vec, dt})[0];
        JdF = Eigen::Map<matrixx_t>(static_cast<std::vector<double>>(JdF_res).data(), vel_dim_, CONTACT_3DOF*num_contacts_);

        // std::cout << "Jdq:\n" << Jdq << std::endl;
        // std::cout << "Jdv:\n" << Jdv << std::endl;
        // std::cout << "Jdtau:\n" << Jdtau << std::endl;
        // std::cout << "JdF:\n" << JdF << std::endl;

        // // Check against C++
        // vectorx_t q_eval = models::ConvertdqToq(dq1, q1_lin);
        // std::cout << "C++ q: " << q_eval.transpose() << std::endl;
        //
        // std::unique_ptr<casadi::Function> casadi_dq_q_function_ = std::make_unique<casadi::Function>(
        //     casadi::external("dq_q", "dynamics_functions.so"));
        //
        // casadi::DM q_res = (*casadi_dq_q_function_)(casadi::DMVector{dq1_vec, q1_vec})[0];
        // vectorx_t q_casadi_vec = Eigen::Map<vectorx_t>(
        //   static_cast<std::vector<double>>(q_res).data(), model_.GetConfigDim(), 1);
        // std::cout << "Casadi q: " << q_casadi_vec.transpose() << std::endl;
        //
        // vectorx_t v_eval = v1_lin + dv1;
        // vectorx_t tau_eval = tau_lin + dtau;
        // vectorx_t force_eval = force_lin + dforce;
        // pinocchio::Data data(model_.GetModel());
        //
        // int idx = 0;
        // std::vector<models::ExternalForce<double>> f_ext;
        // for (const auto& frame : contact_frames_) {
        //     f_ext.emplace_back(frame, force_eval.segment<CONTACT_3DOF>(idx));
        //     idx += CONTACT_3DOF;
        // }
        //
        //
        // vectorx_t a_pin = models::ForwardDynamics(model_.GetModel(), data, q_eval, v_eval, tau_eval, f_ext);
        //
        // std::cout << "CasADi aba:\n" << "\ta = " << a_casadi_vec.transpose() << std::endl;
        // std::cout << "pinocchio aba:\n" << "\ta = " << a_pin.transpose() << std::endl;
        //
        // std::cout << "g1 config dim: " << model_.GetConfigDim() << std::endl;
        // std::cout << "g1 vel dim: " << model_.GetVelDim() << std::endl;
    }

}
