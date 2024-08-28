//
// Created by zolkin on 7/31/24.
//

#ifndef MPC_TEST_CLASS_H
#define MPC_TEST_CLASS_H

#include <catch2/catch_test_macros.hpp>

#include "pinocchio/algorithm/joint-configuration.hpp"

#include "full_order_mpc.h"

namespace torc::mpc {
    class MpcTestClass : public FullOrderMpc {
    public:
        MpcTestClass(const fs::path& config_file, const fs::path& model_path, const std::string& name)
            : FullOrderMpc(name, config_file, model_path) {
            CHECK(dt_.size() == nodes_);
        }

        void CheckQuaternionIntLin() {
            PrintTestHeader("Quaternion Integration Linearization");

            // TODO: Put back to 5
            for (int k = 0; k < 1; k++) {
                // Random state
                vectorx_t q_rand = robot_model_->GetRandomConfig();
                vectorx_t q2_rand = robot_model_->GetRandomConfig();

                vectorx_t v_rand = robot_model_->GetRandomVel();
                vectorx_t v2_rand = robot_model_->GetRandomVel();

                traj_.SetConfiguration(0, q_rand);
                traj_.SetVelocity(0, v_rand);
                traj_.SetConfiguration(1, q2_rand);
                traj_.SetVelocity(1, v2_rand);

                vectorx_t x(config_dim_ + 2*vel_dim_);
                x << traj_.GetConfiguration(0), traj_.GetVelocity(0), traj_.GetVelocity(1);
                vectorx_t p(1);
                p(0) = dt_[0];
                matrixx_t jac_analytic;
                integration_constraint_->GetJacobian(x, p, jac_analytic);

                matrix3_t dxi_ad = jac_analytic.block<3,3>(3, 3);

                // xi
                // Analytic
                matrix3_t dxi = QuatIntegrationLinearizationXi(0);

                // Finite difference
                matrix3_t fd = matrix3_t::Zero();
                vector3_t xi = vector3_t::Zero();
                vector3_t w = 0.5*(traj_.GetVelocity(0).segment<3>(3) + traj_.GetVelocity(1).segment<3>(3));
                vector3_t xi1 = robot_model_->QuaternionIntegrationRelative( traj_.GetQuat(1),
                    traj_.GetQuat(0), xi, w, dt_[0]);
                for (int i = 0; i < 3; i++) {
                    xi(i) += FD_DELTA;
                    vector3_t xi2 = robot_model_->QuaternionIntegrationRelative(traj_.GetQuat(1),
                        traj_.GetQuat(0), xi, w, dt_[0]);
                    fd.col(i) = (xi2 - xi1)/FD_DELTA;

                    xi(i) -= FD_DELTA;
                }
                std::cout << "fd: " << fd << std::endl;
                std::cout << "dxi: " << dxi << std::endl;
                std::cout << "cpp ad: " << dxi_ad << std::endl;
                CHECK(fd.isApprox(dxi_ad, sqrt(FD_DELTA)));

                // w
                // Analytic
                matrix3_t dw = QuatIntegrationLinearizationW(0);
                std::cout << "analytic: " << dw << std::endl;

                // Finite difference
                fd = matrix3_t::Zero();
                xi = vector3_t::Zero();
                for (int i = 0; i < 3; i++) {
                    w(i) += FD_DELTA;
                    vector3_t xi2 = robot_model_->QuaternionIntegrationRelative(traj_.GetQuat(1),
                        traj_.GetQuat(0), xi, w, dt_[0]);
                    fd.col(i) = 0.5*(xi2 - xi1)/FD_DELTA;

                    w(i) -= FD_DELTA;
                }
                CHECK(fd.isApprox(dw, sqrt(FD_DELTA)));
                std::cout << "finite difference: " << fd << std::endl;
            }
        }

        void CheckInverseDynamicsLin() {
            PrintTestHeader("Inverse Dynamics Linearization");

            // Random state
            vectorx_t q_rand = robot_model_->GetRandomConfig();
            vectorx_t q2_rand = robot_model_->GetRandomConfig();

            vectorx_t v_rand = robot_model_->GetRandomVel();
            vectorx_t v2_rand = v_rand + (robot_model_->GetRandomVel() * 0.05);

            dt_[0] = 0.02;;

            std::vector<models::ExternalForce> f_ext;
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

            // Analytic
            matrixx_t dtau_dq, dtau_dv1, dtau_dv2, dtau_df;
            dtau_dq = matrixx_t::Zero(robot_model_->GetVelDim(), robot_model_->GetVelDim());
            dtau_dv1 = matrixx_t::Zero(robot_model_->GetVelDim(), robot_model_->GetVelDim());
            dtau_dv2 = matrixx_t::Zero(robot_model_->GetVelDim(), robot_model_->GetVelDim());
            dtau_df = matrixx_t::Zero(robot_model_->GetVelDim(), num_contact_locations_*3);

            InverseDynamicsLinearization(0, dtau_dq, dtau_dv1, dtau_dv2, dtau_df);

            // Finite Difference
            // ----- Configuration ----- //
            matrixx_t fd_q = matrixx_t::Zero(robot_model_->GetVelDim(), robot_model_->GetVelDim());
            vectorx_t a = (v2_rand - v_rand)/dt_[0];
            vectorx_t tau1 = robot_model_->InverseDynamics(q_rand, v_rand, a, f_ext);
            for (int i = 0; i < robot_model_->GetVelDim(); i++) {
                PerturbConfiguration(q_rand, FD_DELTA, i);
                // std::cout << "q_rand pert: " << q_rand.transpose() << std::endl;
                vectorx_t q = traj_.GetConfiguration(0);
                vectorx_t v_eps = vectorx_t::Zero(robot_model_->GetVelDim());
                v_eps(i) += FD_DELTA;
                // std::cout << "q integrate: " << pinocchio::integrate(robot_model_->GetModel(), q, v_eps).transpose() << std::endl;
                vectorx_t tau2 = robot_model_->InverseDynamics(q_rand, v_rand, a, f_ext);

                fd_q.col(i) = (tau2 - tau1)/FD_DELTA;

                q_rand = traj_.GetConfiguration(0);
            }
            CHECK(dtau_dq.isApprox(fd_q, sqrt(FD_DELTA)));

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

            CHECK(dtau_dv1.isApprox(fd_v, sqrt(FD_DELTA)));

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
            CHECK(dtau_dv2.isApprox(fd_v2, sqrt(FD_DELTA)));

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
            CHECK(dtau_df.isApprox(fd_f, sqrt(FD_DELTA)));
            // std::cout << "f analytic: \n" << dtau_df << std::endl;
            // std::cout << "f fd: \n" << fd_f << std::endl;
        }


        void CheckQuaternionLin() {
            PrintTestHeader("Quaternion Value Linearization");

            for (int k = 0; k < 5; k++) {
                // Random state
                vectorx_t q_rand = robot_model_->GetRandomConfig();
                traj_.SetConfiguration(0, q_rand);

                // Analytic
                matrix43_t q_int_lin = QuatLinearization(0);

                // Finite difference
                matrix43_t fd = matrix43_t::Zero();
                quat_t q1 = traj_.GetQuat(0);
                vector3_t v = vector3_t::Zero();
                for (int i = 0; i < 3; i++) {
                    v(i) += FD_DELTA;
                    quat_t q2 = q1*pinocchio::quaternion::exp3(v);
                    v(i) -= FD_DELTA;

                    fd(0, i) = (q2.x() - q1.x())/FD_DELTA;
                    fd(1, i) = (q2.y() - q1.y())/FD_DELTA;
                    fd(2, i) = (q2.z() - q1.z())/FD_DELTA;
                    fd(3, i) = (q2.w() - q1.w())/FD_DELTA;

                    for (int j = 0; j < 4; j++) {
                        CHECK_THAT(fd(j,i) - q_int_lin(j, i),
                            Catch::Matchers::WithinAbs(0, FD_MARGIN));
                    }
                }
            }
        }

        void CheckSwingHeightLin() {
            PrintTestHeader("Swing Height Linearization");

            for (int k = 0; k < 5; k++) {
                vectorx_t q_rand = robot_model_->GetRandomConfig();
                vectorx_t q_original = q_rand;
                traj_.SetConfiguration(0, q_rand);

                for (const auto& frame : contact_frames_) {
                    // Analytic solution
                    matrix6x_t frame_jacobian = matrix6x_t::Zero(6, robot_model_->GetVelDim());
                    SwingHeightLinearization(0, frame, frame_jacobian);

                    // Finite difference
                    matrix6x_t frame_fd = matrix6x_t::Zero(6, robot_model_->GetVelDim());
                    robot_model_->FirstOrderFK(q_rand);
                    vector3_t frame_pos_nom = robot_model_->GetFrameState(frame).placement.translation();
                    for (int i = 0; i < robot_model_->GetVelDim(); i++) {
                        PerturbConfiguration(q_rand, FD_DELTA, i);
                        robot_model_->FirstOrderFK(q_rand);
                        vector3_t frame_pos_pert = robot_model_->GetFrameState(frame).placement.translation();
                        q_rand = q_original;

                        for (int j = 0; j < frame_pos_nom.size(); j++) {
                            frame_fd(j, i) = (frame_pos_pert(j) - frame_pos_nom(j))/FD_DELTA;
                            CHECK_THAT(frame_fd(j,i) - frame_jacobian(j, i),
                                Catch::Matchers::WithinAbs(0, FD_MARGIN));
                        }
                    }
                }
            }
        }

        void CheckHolonomicLin() {
            PrintTestHeader("Holonomic Linearization");

            // TODO: Put back to 5
            for (int k = 0; k < 1; k++) {
                vectorx_t q_rand = robot_model_->GetRandomConfig();
                vectorx_t v_rand = robot_model_->GetRandomVel();
                vectorx_t q_original = q_rand;
                traj_.SetConfiguration(0, q_rand);
                traj_.SetVelocity(0, v_rand);

                for (const auto& frame : contact_frames_) {
                    // ------- Configuration ------- //
                    // Analytic solution
                    matrix6x_t jacobian = matrix6x_t::Zero(6, robot_model_->GetVelDim());
                    HolonomicLinearizationq(0, frame, jacobian);

                    // Finite difference
                    matrix6x_t frame_fd = matrix6x_t::Zero(6, robot_model_->GetVelDim());
                    vector3_t frame_vel = robot_model_->GetFrameState(frame, q_rand, v_rand, pinocchio::LOCAL_WORLD_ALIGNED).vel.linear();
                    for (int i = 0; i < robot_model_->GetVelDim(); i++) {
                        PerturbConfiguration(q_rand, FD_DELTA, i);
                        vector3_t frame_pos_pert = robot_model_->GetFrameState(frame, q_rand, v_rand, pinocchio::LOCAL_WORLD_ALIGNED).vel.linear();
                        q_rand = q_original;

                        frame_fd.block(0, i, 3, 1) = (frame_pos_pert - frame_vel)/FD_DELTA;

                        // for (int j = 0; j < frame_vel.size(); j++) {
                        //     frame_fd(j, i) = (frame_pos_pert(j) - frame_vel(j))/FD_DELTA;
                        //     CHECK_THAT(frame_fd(j,i) - jacobian(j, i),
                        //         Catch::Matchers::WithinAbs(0, FD_MARGIN));
                        // }
                    }

                    std::cout << "configuration" << std::endl;
                    std::cout << "jac analytic: \n" << jacobian.topRows<3>() << std::endl;
                    std::cout << "jac fd: \n" << frame_fd.topRows<3>() << std::endl << std::endl;
                    CHECK(jacobian.topRows<3>().isApprox(frame_fd.topRows<3>(), sqrt(FD_DELTA)));

                    // ------- Velocity ------- //
                    // Analytic solution
                    jacobian.setZero();
                    HolonomicLinearizationv(0, frame, jacobian);

                    // Finite difference
                    frame_fd.setZero();
                    frame_vel = robot_model_->GetFrameState(frame, q_rand, v_rand, pinocchio::LOCAL_WORLD_ALIGNED).vel.linear();
                    for (int i = 0; i < robot_model_->GetVelDim(); i++) {
                        v_rand(i) += FD_DELTA;
                        vector3_t frame_pos_pert = robot_model_->GetFrameState(frame, q_rand, v_rand, pinocchio::LOCAL_WORLD_ALIGNED).vel.linear();
                        v_rand(i) -= FD_DELTA;

                        frame_fd.block(0, i, 3, 1) = (frame_pos_pert - frame_vel)/FD_DELTA;

                        // for (int j = 0; j < frame_vel.size(); j++) {
                        //     frame_fd(j, i) = (frame_pos_pert(j) - frame_vel(j))/FD_DELTA;
                        //     // CHECK_THAT(frame_fd(j,i) - jacobian(j, i),
                        //     //     Catch::Matchers::WithinAbs(0, FD_MARGIN));
                        // }
                    }

                    CHECK(jacobian.topRows<3>().isApprox(frame_fd.topRows<3>(), sqrt(FD_DELTA)));

                    std::cout << "velocity" << std::endl;
                    std::cout << "jac analytic: \n" << jacobian.topRows<3>() << std::endl;
                    std::cout << "jac fd: \n" << frame_fd.topRows<3>() << std::endl << std::endl;
                }
            }
        }

        void CheckCostFunctionDefiniteness() {
            PrintTestHeader("Cost Function Definiteness");

            for (int k = 0; k < 5; k++) {
                traj_.SetNumNodes(nodes_);
                q_target_.resize(nodes_);
                v_target_.resize(nodes_);
                for (int i = 0; i < nodes_; i++) {
                    traj_.SetConfiguration(i, robot_model_->GetRandomConfig());
                    q_target_[i] = robot_model_->GetRandomConfig();
                    traj_.SetVelocity(i, robot_model_->GetRandomVel());
                    v_target_[i] = robot_model_->GetRandomVel();
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
            int row1 = 2*robot_model_->GetVelDim();
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
            }

            CHECK(row1 == GetConstraintRowStartNode(nodes_));
        }


        void CheckDefaultSwingTraj() {
            PrintTestHeader("Default Swing Traj.");
            ContactSchedule cs;
            cs.SetFrames(contact_frames_);
            cs.InsertContact(contact_frames_[0], 0.3, 0.6);
            UpdateContactScheduleAndSwingTraj(cs, 1, 0, 0.5);
            for (int i = 0; i < nodes_; i++) {
                std::cout << swing_traj_[contact_frames_[0]][i] << std::endl;
            }
            CHECK(swing_traj_[contact_frames_[0]][0] == 0.0);
            std::cout << "====" << std::endl;

            // At time zero we should be at the apex
            cs.InsertContact(contact_frames_[0], -0.2, -0.1);
            cs.InsertContact(contact_frames_[0], 0.1, 0.15);

            UpdateContactScheduleAndSwingTraj(cs, 1, 0, 0.5);
            for (int i = 0; i < nodes_; i++) {
                std::cout << swing_traj_[contact_frames_[0]][i] << std::endl;
            }
            CHECK(swing_traj_[contact_frames_[0]][0] == 1.0);
        }

        // ---------------------- //
        // ----- Benchmarks ----- //
        // ---------------------- //
        void BenchmarkQuaternionIntegrationLin() {
            BENCHMARK("quaternion integration lin") {
                auto deriv = QuatIntegrationLinearizationXi(0);
            };
        }

        void BenchmarkInverseDynamicsLin() {
            // Random state
            vectorx_t q_rand = robot_model_->GetRandomConfig();
            vectorx_t q2_rand = robot_model_->GetRandomConfig();

            vectorx_t v_rand = robot_model_->GetRandomVel();
            vectorx_t v2_rand = v_rand + (robot_model_->GetRandomVel() * 0.05);

            dt_[0] = 0.02;

            std::vector<models::ExternalForce> f_ext;
            for (const auto& frame : contact_frames_) {
                vector3_t force = vector3_t::Random().cwiseMax(-100).cwiseMin(100); //vector3_t::Zero();
                std::cout << "force: " << force.transpose() << std::endl;
                traj_.SetForce(0, frame, force);
                f_ext.emplace_back(frame, force);
            }

            traj_.SetConfiguration(0, q_rand);
            traj_.SetConfiguration(1, q2_rand);
            traj_.SetVelocity(0, v_rand);
            traj_.SetVelocity(1, v2_rand);

            // Analytic
            matrixx_t dtau_dq, dtau_dv1, dtau_dv2, dtau_df;
            dtau_dq = matrixx_t::Zero(robot_model_->GetVelDim(), robot_model_->GetVelDim());
            dtau_dv1 = matrixx_t::Zero(robot_model_->GetVelDim(), robot_model_->GetVelDim());
            dtau_dv2 = matrixx_t::Zero(robot_model_->GetVelDim(), robot_model_->GetVelDim());
            dtau_df = matrixx_t::Zero(robot_model_->GetVelDim(), num_contact_locations_*3);

            BENCHMARK("inverse dynamics lin") {
                InverseDynamicsLinearization(0, dtau_dq, dtau_dv1, dtau_dv2, dtau_df);
            };

        }

        void BenchmarkQuaternionConfigurationLin() {
            BENCHMARK("quaternion configuration lin") {
                auto deriv = QuatLinearization(0);
            };
        }

        void BenchmarkSwingHeightLin() {
            vectorx_t q_rand = robot_model_->GetRandomConfig();
            traj_.SetConfiguration(0, q_rand);
            matrix6x_t frame_jacobian = matrix6x_t::Zero(6, robot_model_->GetVelDim());
            std::string frame = contact_frames_[0];
            BENCHMARK("swing height lin") {
                SwingHeightLinearization(0, frame, frame_jacobian);
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
                HolonomicLinearizationq(0, frame, jacobian);
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
