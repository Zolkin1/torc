//
// Created by zolkin on 7/31/24.
//

#ifndef MPC_TEST_CLASS_H
#define MPC_TEST_CLASS_H

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "pinocchio/algorithm/joint-configuration.hpp"

#include "full_order_mpc.h"

namespace torc::mpc {
    class MpcTestClass : public FullOrderMpc {
    public:
        MpcTestClass(const fs::path& config_file, const fs::path& model_path, const std::string& name)
            : FullOrderMpc(name, config_file, model_path) {
            CHECK(dt_.size() == nodes_);

            CHECK(NumIntegratorConstraintsNode() == integration_constraint_->GetRangeSize());
            CHECK(NumHolonomicConstraintsNode() == holonomic_constraint_[contact_frames_[0]]->GetRangeSize()*num_contact_locations_);
            CHECK(NumSwingHeightConstraintsNode() == swing_height_constraint_[contact_frames_[0]]->GetRangeSize()*num_contact_locations_);
        }

        // TODO: Understand!
        void CheckPinIntegrate() {
        // TODO: Understand!!
            PrintTestHeader("Pinocchio Integrate");
            vectorx_t q_rand = robot_model_->GetRandomConfig();
            vectorx_t v_rand = robot_model_->GetRandomVel();
            v_rand.segment<3>(POS_VARS).setZero();
            // v_rand.head<3>().setZero();

            std::cout << "q_rand floating base: " << q_rand.head<FLOATING_BASE>().transpose() << std::endl;
            std::cout << "v_rand floating base: " << v_rand.head<FLOATING_VEL>().transpose() << std::endl;

            vectorx_t q = pinocchio::integrate(robot_model_->GetModel(), q_rand, v_rand);

            std::cout << "q_pin floating base: " << q.head<FLOATING_BASE>().transpose() << std::endl;

            vector3_t virtual_pos = q_rand.head<POS_VARS>() + v_rand.head<POS_VARS>();
            matrix3_t T = matrix3_t::Zero();
            T(0, 1) = -virtual_pos(2);
            T(0, 2) = virtual_pos(1);
            T(1, 0) = virtual_pos(2);
            T(1, 2) = -virtual_pos(0);
            T(2, 0) = -virtual_pos(1);
            T(2, 1) = virtual_pos(0);

            // std::cout << "T: \n" << T << std::endl;

            Eigen::Quaterniond quat(q_rand.segment<QUAT_VARS>(POS_VARS));
            Eigen::Quaterniond quat2(q_rand.segment<QUAT_VARS>(POS_VARS));
            matrix3_t R = quat2.toRotationMatrix();     // Maps from the body frame to the world frame
            vectorx_t q_manual(FLOATING_BASE);
            q_manual << q_rand.head<POS_VARS>() + R*v_rand.head<POS_VARS>(), // + T*R*v_rand.segment<3>(POS_VARS),
                            (quat * pinocchio::quaternion::exp3(v_rand.segment<3>(POS_VARS))).coeffs();

            std::cout << "q_man floating base: " << q_manual.transpose() << std::endl;

            std::cout << "R: \n" << R << std::endl;
        }

        void CheckSwingHeightConstraint() {
            PrintTestHeader("Swing Height Constraint");
            static constexpr double MARGIN = 1e-6;
            const int node = 0;

            for (int k = 0; k < 5; k++) {
                vectorx_t q_rand = robot_model_->GetRandomConfig();

                traj_.SetConfiguration(node, q_rand);

                robot_model_->FirstOrderFK(q_rand);
                for (const auto& frame : contact_frames_) {
                    std::cout << "Frame " << frame << std::endl;
                    swing_traj_[frame][node] = 0.1;
                    std::cout << "Desired swing height: " << swing_traj_[frame][node] << std::endl;

                    vectorx_t x_zero = vectorx_t::Zero(swing_height_constraint_[frame]->GetDomainSize());
                    vectorx_t p(swing_height_constraint_[frame]->GetParameterSize());
                    p << traj_.GetConfiguration(node), swing_traj_[frame][node];

                    vector3_t frame_pos = robot_model_->GetFrameState(frame).placement.translation();
                    std::cout << "frame pos: " << frame_pos.transpose() << std::endl;

                    vectorx_t y;
                    swing_height_constraint_[frame]->GetFunctionValue(x_zero, p, y);
                    std::cout << "constraint violation: " << y(0) << std::endl;

                    CHECK_THAT(frame_pos(2) - swing_traj_[frame][node], Catch::Matchers::WithinAbs(y(0), MARGIN));

                    std::cout << std::endl;
                }
            }
        }

        // TODO: Fix for ad derivative
        // void CheckQuaternionIntLin() {
        //     PrintTestHeader("Quaternion Integration Linearization");
        //
        //     // TODO: Put back to 5
        //     for (int k = 0; k < 1; k++) {
        //         // Random state
        //         vectorx_t q_rand = robot_model_->GetRandomConfig();
        //         vectorx_t q2_rand = robot_model_->GetRandomConfig();
        //
        //         vectorx_t v_rand = robot_model_->GetRandomVel();
        //         vectorx_t v2_rand = robot_model_->GetRandomVel();
        //
        //         traj_.SetConfiguration(0, q_rand);
        //         traj_.SetVelocity(0, v_rand);
        //         traj_.SetConfiguration(1, q2_rand);
        //         traj_.SetVelocity(1, v2_rand);
        //
        //         vectorx_t x(4*vel_dim_);
        //         vectorx_t dq_zero = vectorx_t::Zero(vel_dim_);
        //         vectorx_t dv_zero = vectorx_t::Zero(vel_dim_);
        //
        //         x << dq_zero, dq_zero, dv_zero, dv_zero; //traj_.GetVelocity(0), traj_.GetVelocity(1);
        //
        //         vectorx_t p(1 + 2*config_dim_ + 2*vel_dim_);
        //         p << dt_[0], traj_.GetConfiguration(0), traj_.GetConfiguration(1), traj_.GetVelocity(0), traj_.GetVelocity(1);
        //         matrixx_t jac_analytic;
        //         integration_constraint_->GetJacobian(x, p, jac_analytic);
        //
        //         std::cout << "cpp ad full: \n" << jac_analytic << std::endl;
        //
        //         matrix3_t dxi_ad = jac_analytic.block<3,3>(3, 3);
        //
        //         // xi
        //         // Analytic
        //         matrix3_t dxi = QuatIntegrationLinearizationXi(0);
        //
        //         // Finite difference
        //         matrix3_t fd = matrix3_t::Zero();
        //         vector3_t xi = vector3_t::Zero();
        //         vector3_t w = 0.5*(traj_.GetVelocity(0).segment<3>(3) + traj_.GetVelocity(1).segment<3>(3));
        //         vector3_t xi1 = robot_model_->QuaternionIntegrationRelative( traj_.GetQuat(1),
        //             traj_.GetQuat(0), xi, w, dt_[0]);
        //         for (int i = 0; i < 3; i++) {
        //             xi(i) += FD_DELTA;
        //             vector3_t xi2 = robot_model_->QuaternionIntegrationRelative(traj_.GetQuat(1),
        //                 traj_.GetQuat(0), xi, w, dt_[0]);
        //             fd.col(i) = (xi2 - xi1)/FD_DELTA;
        //
        //             xi(i) -= FD_DELTA;
        //         }
        //         std::cout << "fd: \n" << fd << std::endl;
        //         std::cout << "dxi: \n" << dxi << std::endl;
        //         std::cout << "cpp ad: \n" << dxi_ad << std::endl;
        //         CHECK(fd.isApprox(dxi_ad, sqrt(FD_DELTA)));
        //
        //         // w
        //         // Analytic
        //         matrix3_t dw = QuatIntegrationLinearizationW(0);
        //         std::cout << "analytic: " << dw << std::endl;
        //
        //         // Finite difference
        //         fd = matrix3_t::Zero();
        //         xi = vector3_t::Zero();
        //         for (int i = 0; i < 3; i++) {
        //             w(i) += FD_DELTA;
        //             vector3_t xi2 = robot_model_->QuaternionIntegrationRelative(traj_.GetQuat(1),
        //                 traj_.GetQuat(0), xi, w, dt_[0]);
        //             fd.col(i) = 0.5*(xi2 - xi1)/FD_DELTA;
        //
        //             w(i) -= FD_DELTA;
        //         }
        //         CHECK(fd.isApprox(dw, sqrt(FD_DELTA)));
        //         std::cout << "finite difference: " << fd << std::endl;
        //     }
        // }

        void CheckInverseDynamicsLin() {
            PrintTestHeader("Inverse Dynamics Linearization");

            // Random state
            vectorx_t q_rand = robot_model_->GetRandomConfig();
            vectorx_t q2_rand = robot_model_->GetRandomConfig();

            vectorx_t v_rand = robot_model_->GetRandomVel();
            vectorx_t v2_rand = v_rand + (robot_model_->GetRandomVel() * 0.05);

            dt_[0] = 0.02;;

            std::vector<models::ExternalForce<double>> f_ext;
            for (const auto& frame : contact_frames_) {
                vector3_t force = vector3_t::Random().cwiseMax(-100).cwiseMin(100); //vector3_t::Zero();
                // std::cout << "force: " << force.transpose() << std::endl;
                traj_.SetForce(0, frame, force);
                f_ext.emplace_back(frame, force);
            }

            traj_.SetConfiguration(0, q_rand);
            traj_.SetConfiguration(1, q2_rand);
            traj_.SetVelocity(0, v_rand);
            traj_.SetVelocity(1, v2_rand);

            // AD
            matrixx_t dtau_dq_ad, dtau_dv1_ad, dtau_dv2_ad, dtau_df_ad, dtau;
            dtau_dq_ad = matrixx_t::Zero(robot_model_->GetVelDim(), robot_model_->GetVelDim());
            dtau_dv1_ad = matrixx_t::Zero(robot_model_->GetVelDim(), robot_model_->GetVelDim());
            dtau_dv2_ad = matrixx_t::Zero(robot_model_->GetVelDim(), robot_model_->GetVelDim());
            dtau_df_ad = matrixx_t::Zero(robot_model_->GetVelDim(), num_contact_locations_*3);

            vectorx_t y;

            InverseDynamicsLinearizationAD(0, dtau_dq_ad, dtau_dv1_ad, dtau_dv2_ad, dtau_df_ad, dtau, y);

            // Finite Difference
            // ----- Configuration ----- //
            matrixx_t fd_q = matrixx_t::Zero(robot_model_->GetVelDim(), robot_model_->GetVelDim());
            vectorx_t a = (v2_rand - v_rand)/dt_[0];
            vectorx_t tau1 = robot_model_->InverseDynamics(q_rand, v_rand, a, f_ext);
            for (int i = 0; i < robot_model_->GetVelDim(); i++) {
                PerturbConfiguration(q_rand, FD_DELTA, i);
                vectorx_t q = traj_.GetConfiguration(0);
                vectorx_t v_eps = vectorx_t::Zero(robot_model_->GetVelDim());
                v_eps(i) += FD_DELTA;
                vectorx_t tau2 = robot_model_->InverseDynamics(q_rand, v_rand, a, f_ext);

                fd_q.col(i) = (tau2 - tau1)/FD_DELTA;

                q_rand = traj_.GetConfiguration(0);
            }
            CHECK(dtau_dq_ad.isApprox(fd_q, sqrt(FD_DELTA)));

            // std::cout << "q analytic: \n" << dtau_dq << std::endl;
            // std::cout << "q fd: \n" << fd_q << std::endl;

            // ----- Velocity ----- //
            matrixx_t fd_v = matrixx_t::Zero(robot_model_->GetVelDim(), robot_model_->GetVelDim());
            for (int i = 0; i < robot_model_->GetVelDim(); i++) {
                v_rand(i) += FD_DELTA;
                a = (v2_rand - v_rand)/dt_[0];

                vectorx_t tau2 = robot_model_->InverseDynamics(q_rand, v_rand, a, f_ext);

                fd_v.col(i) = (tau2 - tau1)/FD_DELTA;

                v_rand(i) -= FD_DELTA;
            }

            CHECK(dtau_dv1_ad.isApprox(fd_v, sqrt(FD_DELTA)));

            // std::cout << "v analytic: \n" << dtau_dv1 << std::endl;
            // std::cout << "v fd: \n" << fd_v << std::endl;

            // ----- Velocity 2 ----- //
            matrixx_t fd_v2 = matrixx_t::Zero(robot_model_->GetVelDim(), robot_model_->GetVelDim());
            for (int i = 0; i < robot_model_->GetVelDim(); i++) {
                v2_rand(i) += FD_DELTA;
                a = (v2_rand - v_rand)/dt_[0];
                vectorx_t tau2 = robot_model_->InverseDynamics(q_rand, v_rand, a, f_ext);

                fd_v2.col(i) = (tau2 - tau1)/FD_DELTA;

                v2_rand(i) -= FD_DELTA;
            }
            CHECK(dtau_dv2_ad.isApprox(fd_v2, sqrt(FD_DELTA)));

            // std::cout << "v2 analytic: \n" << dtau_dv2 << std::endl;
            // std::cout << "v2 fd: \n" << fd_v2 << std::endl;

            a = (v2_rand - v_rand)/dt_[0];

            // ----- Forces ----- //
            matrixx_t fd_f = matrixx_t::Zero(robot_model_->GetVelDim(), num_contact_locations_*CONTACT_3DOF);
            for (int i = 0; i < num_contact_locations_; i++) {
                for (int j = 0; j < CONTACT_3DOF; j++) {
                    f_ext[i].force_linear(j) += FD_DELTA;
                    vectorx_t tau2 = robot_model_->InverseDynamics(q_rand, v_rand, a, f_ext);
                    f_ext[i].force_linear(j) -= FD_DELTA;
                    fd_f.col(i*CONTACT_3DOF + j) = (tau2 - tau1)/FD_DELTA;
                }
            }
            CHECK(dtau_df_ad.isApprox(fd_f, sqrt(FD_DELTA)));
            // std::cout << "f analytic: \n" << dtau_df << std::endl;
            // std::cout << "f fd: \n" << fd_f << std::endl;
        }

        void CheckSwingHeightLin() {
            PrintTestHeader("Swing Height Linearization");

            for (int k = 0; k < 5; k++) {
                vectorx_t q_rand = robot_model_->GetRandomConfig();

                for (const auto& frame : contact_frames_) {
                    // Autodiff
                    matrixx_t jac;
                    vectorx_t x_zero = vectorx_t::Zero(swing_height_constraint_[frame]->GetDomainSize());
                    vectorx_t p(swing_height_constraint_[frame]->GetParameterSize());
                    p << q_rand, 0.08;
                    swing_height_constraint_[frame]->GetJacobian(x_zero, p, jac);

                    // Finite difference
                    vectorx_t y = vectorx_t::Zero(swing_height_constraint_[frame]->GetRangeSize());
                    swing_height_constraint_[frame]->GetFunctionValue(x_zero, p, y);

                    matrixx_t jac_fd = matrixx_t::Zero(1, robot_model_->GetVelDim());
                    vectorx_t y_pert = vectorx_t::Zero(swing_height_constraint_[frame]->GetRangeSize());
                    for (int i = 0; i < robot_model_->GetVelDim(); i++) {
                        x_zero(i) = FD_DELTA;
                        swing_height_constraint_[frame]->GetFunctionValue(x_zero, p, y_pert);
                        x_zero(i) = 0;

                        jac_fd(0, i) = (-y(0) + y_pert(0))/FD_DELTA;
                    }
                    std::cout << "jac:\n" << jac << "\nfd:\n" << jac_fd << std::endl;
                    CHECK(jac.isApprox(jac_fd, 1e-5));
                }
            }
        }

        void CheckHolonomicLin() {
            PrintTestHeader("Holonomic Linearization");

            for (int k = 0; k < 5; k++) {
                vectorx_t q_rand = robot_model_->GetRandomConfig();
                vectorx_t v_rand = robot_model_->GetRandomVel();

                for (const auto& frame : contact_frames_) {
                    //Autodiff
                    matrixx_t jac;
                    vectorx_t x_zero = vectorx_t::Zero(holonomic_constraint_[frame]->GetDomainSize());
                    vectorx_t p(holonomic_constraint_[frame]->GetParameterSize());
                    p << q_rand, v_rand;

                    holonomic_constraint_[frame]->GetJacobian(x_zero, p, jac);

                    // Finite difference
                    vectorx_t y = vectorx_t::Zero(holonomic_constraint_[frame]->GetRangeSize());
                    holonomic_constraint_[frame]->GetFunctionValue(x_zero, p, y);

                    matrixx_t jac_fd = matrixx_t::Zero(holonomic_constraint_[frame]->GetRangeSize(), holonomic_constraint_[frame]->GetDomainSize());
                    vectorx_t y_pert = vectorx_t::Zero(holonomic_constraint_[frame]->GetRangeSize());
                    for (int i = 0; i < 2*robot_model_->GetVelDim(); i++) {
                        x_zero(i) = FD_DELTA;
                        holonomic_constraint_[frame]->GetFunctionValue(x_zero, p, y_pert);
                        x_zero(i) = 0;

                        jac_fd.col(i) = (y_pert - y)/FD_DELTA;
                    }

                    std::cout << "jac autodiff:\n" << jac<< std::endl;
                    std::cout << "jac fd:\n" << jac_fd << std::endl << std::endl;
                    CHECK(jac.isApprox(jac_fd, sqrt(FD_DELTA)));
                }
            }
        }

        void CheckCostFunctionDefiniteness() {
            PrintTestHeader("Cost Function Definiteness");

            for (int k = 0; k < 5; k++) {
                traj_.SetNumNodes(nodes_);
                for (int i = 0; i < nodes_; i++) {
                    traj_.SetConfiguration(i, robot_model_->GetRandomConfig());
                    q_target_.InsertData(i, robot_model_->GetRandomConfig());
                    traj_.SetVelocity(i, robot_model_->GetRandomVel());
                    v_target_.InsertData(i, robot_model_->GetRandomVel());
                    traj_.SetTau(i, vectorx_t::Random(robot_model_->GetNumInputs()));

                    for (const auto& frame : contact_frames_) {
                        vector3_t force = vector3_t::Random().cwiseMax(-100).cwiseMin(100); //vector3_t::Zero();
                        // std::cout << "force: " << force.transpose() << std::endl;
                        traj_.SetForce(i, frame, force);
                    }
                }

                UpdateCost();

                for (const auto& objective_triplet : objective_triplets_) {
                    REQUIRE(!std::isnan(objective_triplet.value()));
                }

                for (const auto& val : osqp_instance_.objective_vector) {
                    REQUIRE(!std::isnan(val));
                }

                CHECK(objective_mat_.isApprox(objective_mat_.transpose()));

                Eigen::LDLT<matrixx_t> ldlt(objective_mat_);
                CHECK(ldlt.info() != Eigen::NumericalIssue);
                CHECK(ldlt.isPositive());
            }
        }

        void CheckConstraintIdx() {
            PrintTestHeader("Constraint Index");
            int row1 = 0; //2*robot_model_->GetVelDim();
            int row2 = 0;

            for (int node = 0; node < nodes_; node++) {

                CHECK(GetConstraintRowStartNode(node) == row1);

                if (node < nodes_ - 1) {
                    row2 = GetConstraintRow(node, Integrator);
                    CHECK(row2 == row1);
                    row1 += NumIntegratorConstraintsNode();
                }

                if (node < nodes_full_dynamics_) {
                    row2 = GetConstraintRow(node, ID);
                    CHECK(row2 == row1);
                    row1 += NumIDConstraintsNode();
                } else if (node < nodes_ - 1) {
                    row2 = GetConstraintRow(node, ID);
                    CHECK(row2 == row1);
                    row1 += NumPartialIDConstraintsNode();
                }

                row2 = GetConstraintRow(node, FrictionCone);
                CHECK(row2 == row1);
                row1 += NumFrictionConeConstraintsNode();

                if (node > 0) {
                    row2 = GetConstraintRow(node, ConfigBox);
                    CHECK(row2 == row1);
                    row1 += NumConfigBoxConstraintsNode();
                }

                if (node > 0) {
                    row2 = GetConstraintRow(node, VelBox);
                    CHECK(row2 == row1);
                    row1 += NumVelocityBoxConstraintsNode();
                }

                if (node < nodes_full_dynamics_) {
                    row2 = GetConstraintRow(node, TorqueBox);
                    CHECK(row2 == row1);
                    row1 += NumTorqueBoxConstraintsNode();
                }

                if (node > 0) {
                    row2 = GetConstraintRow(node, SwingHeight);
                    CHECK(row2 == row1);
                    row1 += NumSwingHeightConstraintsNode();
                }

                if (node > 0) {
                    row2 = GetConstraintRow(node, Holonomic);
                    CHECK(row2 == row1);
                    row1 += NumHolonomicConstraintsNode();
                }

                if (node > 0) {
                    row2 = GetConstraintRow(node, Collision);
                    CHECK(row2 == row1);
                    row1 += NumCollisionConstraintsNode();
                }

                if (node > 0) {
                    row2 = GetConstraintRow(node, FootPolytope);
                    CHECK(row2 == row1);
                    row1 += NumFootPolytopeConstraintsNode();
                }
            }

            CHECK(row1 == GetConstraintRowStartNode(nodes_));
        }

        void CheckDefaultSwingTraj() {
            PrintTestHeader("Default Swing Traj.");
            ContactSchedule cs;
            cs.SetFrames(contact_frames_);
            cs.InsertSwing(contact_frames_[0], 0.3, 0.6);
            std::vector<double> end_heights(num_contact_locations_);
            for (auto& h : end_heights) {
                h = 0;
            }
            UpdateContactScheduleAndSwingTraj(cs, 1, end_heights, 0.5);
            for (int i = 0; i < nodes_; i++) {
                std::cout << swing_traj_[contact_frames_[0]][i] << std::endl;
            }
            CHECK(swing_traj_[contact_frames_[0]][0] == 0.0);
            std::cout << "====" << std::endl;

            // At time zero we should be at the apex
            cs.InsertSwing(contact_frames_[0], -0.1, 0.1);
            for (auto& h : end_heights) {
                h = 0.01;
            }
            UpdateContactScheduleAndSwingTraj(cs, 1, end_heights, 0.5);
            for (int i = 0; i < nodes_; i++) {
                std::cout << swing_traj_[contact_frames_[0]][i] << std::endl;
            }
            CHECK(swing_traj_[contact_frames_[0]][0] == 1.0);
        }

        // ---------------------- //
        // ----- Benchmarks ----- //
        // ---------------------- //
        void BenchmarkInverseDynamicsLin() {
            // Random state
            vectorx_t q_rand = robot_model_->GetRandomConfig();
            vectorx_t q2_rand = robot_model_->GetRandomConfig();

            vectorx_t v_rand = robot_model_->GetRandomVel();
            vectorx_t v2_rand = v_rand + (robot_model_->GetRandomVel() * 0.05);

            vectorx_t f(CONTACT_3DOF*num_contact_locations_);

            dt_[0] = 0.02;

            int f_idx = 0;
            std::vector<models::ExternalForce<double>> f_ext;
            for (const auto& frame : contact_frames_) {
                vector3_t force = vector3_t::Random().cwiseMax(-100).cwiseMin(100); //vector3_t::Zero();
                f.segment(f_idx, CONTACT_3DOF) = force;
                f_idx += CONTACT_3DOF;
                std::cout << "force: " << force.transpose() << std::endl;
                traj_.SetForce(0, frame, force);
                f_ext.emplace_back(frame, force);
            }

            traj_.SetConfiguration(0, q_rand);
            traj_.SetConfiguration(1, q2_rand);
            traj_.SetVelocity(0, v_rand);
            traj_.SetVelocity(1, v2_rand);

            // Analytic
            matrixx_t dtau_dq, dtau_dv1, dtau_dv2, dtau_df, dtau;
            dtau_dq = matrixx_t::Zero(robot_model_->GetVelDim(), robot_model_->GetVelDim());
            dtau_dv1 = matrixx_t::Zero(robot_model_->GetVelDim(), robot_model_->GetVelDim());
            dtau_dv2 = matrixx_t::Zero(robot_model_->GetVelDim(), robot_model_->GetVelDim());
            dtau_df = matrixx_t::Zero(robot_model_->GetVelDim(), num_contact_locations_*3);

            //BENCHMARK("inverse dynamics lin") {
            //    InverseDynamicsLinearizationAnalytic(0, dtau_dq, dtau_dv1, dtau_dv2, dtau_df);
            //};

            // Autodiff
            vectorx_t tau_temp(input_dim_);
            tau_temp = vectorx_t::Ones(input_dim_);
            traj_.SetTau(0, tau_temp);

            vectorx_t x = vectorx_t::Zero(inverse_dynamics_constraint_->GetDomainSize());
            vectorx_t p(inverse_dynamics_constraint_->GetParameterSize());
            p << traj_.GetConfiguration(0), traj_.GetVelocity(0), traj_.GetVelocity(1), tau_temp, f;
            matrixx_t jac;

            vectorx_t y;

            BENCHMARK("inverse dynamics lin and function call AD") {
                InverseDynamicsLinearizationAD(0, dtau_dq, dtau_dv1, dtau_dv2, dtau_df, dtau, y);
                // inverse_dynamics_constraint_->GetJacobian(x, p, jac);
            };

        }

        void BenchmarkSwingHeightLin() {
            vectorx_t q_rand = robot_model_->GetRandomConfig();
            traj_.SetConfiguration(0, q_rand);
            matrix6x_t frame_jacobian = matrix6x_t::Zero(6, robot_model_->GetVelDim());
            std::string frame = contact_frames_[0];
            BENCHMARK("swing height lin") {
                // SwingHeightLinearization(0, frame, frame_jacobian);
            };
        }

        void BenchmarkHolonomicLin() {
            vectorx_t q_rand = robot_model_->GetRandomConfig();
            vectorx_t v_rand = robot_model_->GetRandomVel();
            traj_.SetConfiguration(0, q_rand);
            traj_.SetVelocity(0, v_rand);
            std::string frame = contact_frames_[0];
            matrix6x_t jacobian = matrix6x_t::Zero(6, robot_model_->GetVelDim());
            BENCHMARK("holonomic lin") {
                // HolonomicLinearizationq(0, frame, jacobian);
            };
        }

        void BenchmarkConstraints() {
            Configure();
            BENCHMARK("mpc add constraints") {
                CreateConstraints();
            };
        }

        void BenchmarkCompute() {
            vectorx_t q_rand = robot_model_->GetRandomConfig();
            vectorx_t v_rand = robot_model_->GetRandomVel();
            Trajectory traj;
            traj.UpdateSizes(robot_model_->GetConfigDim(), robot_model_->GetVelDim(), robot_model_->GetNumInputs(), contact_frames_, nodes_);
            BENCHMARK("mpc compute") {
                Compute(q_rand, v_rand, traj);
            };
        }

        void BenchmarkCostFunctions() {
            // ----- Configuration cost ----- //
            vectorx_t d_rand = robot_model_->GetRandomVel();
            vectorx_t bar_rand = robot_model_->GetRandomConfig();
            vectorx_t target_rand = robot_model_->GetRandomConfig();

            // Analytic
            vectorx_t arg;
            // FormCostFcnArg(d_rand, bar_rand, target_rand, arg);
            // BENCHMARK("configuration cost function gradient") {
            //     vectorx_t grad_c = cost_->Gradient(arg);
            // };

            // BENCHMARK("configuration cost function evaluation") {
            //     double c1 = config_cost_fcn_->Evaluate(arg);
            // };

            // ----- Velocity cost ----- //
            vectorx_t vbar_rand = robot_model_->GetRandomVel();
            vectorx_t vtarget_rand = robot_model_->GetRandomVel();

            // Analytic
            // FormCostFcnArg(d_rand, vbar_rand, vtarget_rand, arg);
            // BENCHMARK("velocity cost function gradient") {
            //     vectorx_t grad_v = vel_cost_fcn_->Gradient(arg);
            // };

            // BENCHMARK("configuration cost function evaluation") {
            //     double c1 = vel_cost_fcn_->Evaluate(arg);
            // };
        }

    protected:
    private:
        void PrintTestHeader(const std::string& name) {
            using std::setw;
            using std::setfill;

            const int total_width = 50;
            std::cout << setfill('=') << setw(total_width/2 - name.size()/2) << " " << name << " " << setw(total_width/2 - name.size()/2) << "" << std::endl;
        }

        // TODO: This is just pinocchio::integrate(robot_model_->GetModel(), q, v_eps)
        void PerturbConfiguration(vectorx_t& q, double delta, int idx) {
            if (idx > robot_model_->GetVelDim()) {
                throw std::runtime_error("Invalid q perturbation index!");
            }

            if (idx < 3) {
                q(idx) += delta;
            } else if (idx < 6) {
                vector3_t q_pert = vector3_t::Zero();
                q_pert(idx - 3) += delta;
                q.array().segment<4>(3) = (static_cast<quat_t>(q.segment<4>(3)) * pinocchio::quaternion::exp3(q_pert)).coeffs();
            } else {
                q(idx + 1) += delta;
            }
        }

        static constexpr double FD_MARGIN = 1e-5;
    };
} // namespacre torc::mpc

#endif //MPC_TEST_CLASS_H
