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
        MpcTestClass(const fs::path& config_file, const fs::path& model_path)
            : FullOrderMpc(config_file, model_path) {}

        void CheckQuaternionIntLin() {
            PrintTestHeader("Quaternion Integration Linearization");

            for (int k = 0; k < 5; k++) {
                // Random state
                vectorx_t q_rand = robot_model_->GetRandomConfig();
                vectorx_t q2_rand = robot_model_->GetRandomConfig();

                vectorx_t v_rand = robot_model_->GetRandomVel();
                vectorx_t v2_rand = robot_model_->GetRandomVel();

                traj_.SetConfiguration(0, q_rand);
                traj_.SetVelocity(0, v_rand);
                traj_.SetConfiguration(1, q2_rand);
                traj_.SetVelocity(1, v2_rand);

                // xi
                // Analytic
                matrix3_t dxi = QuatIntegrationLinearizationXi(0);

                // Finite difference
                matrix3_t fd = matrix3_t::Zero();
                vector3_t xi = vector3_t::Zero();
                vector3_t xi1 = robot_model_->QuaternionIntegrationRelative( traj_.GetQuat(1),
                    traj_.GetQuat(0), xi, traj_.GetVelocity(0).segment<3>(3), 0.02);
                for (int i = 0; i < 3; i++) {
                    xi(i) += FD_DELTA;
                    vector3_t xi2 = robot_model_->QuaternionIntegrationRelative(traj_.GetQuat(1),
                        traj_.GetQuat(0), xi, traj_.GetVelocity(0).segment<3>(3), 0.02);
                    fd.col(i) = (xi2 - xi1)/FD_DELTA;

                    xi(i) -= FD_DELTA;
                }
                CHECK(fd.isApprox(dxi, sqrt(FD_DELTA)));

                // w
                // Analytic
                matrix3_t dw = QuatIntegrationLinearizationW(0);

                // Finite difference
                fd = matrix3_t::Zero();
                xi = vector3_t::Zero();
                vector3_t w = traj_.GetVelocity(0).segment<3>(3);
                for (int i = 0; i < 3; i++) {
                    w(i) += FD_DELTA;
                    vector3_t xi2 = robot_model_->QuaternionIntegrationRelative(traj_.GetQuat(1),
                        traj_.GetQuat(0), xi, w, 0.02);
                    for (int j = 0; j < 3; j++) {
                        fd(j, i) = (xi2(j) - xi1(j))/FD_DELTA;
                        CHECK_THAT(fd(j,i) - dw(j, i),
                            Catch::Matchers::WithinAbs(0, FD_MARGIN));
                    }

                    w(i) -= FD_DELTA;
                }
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

            for (int k = 0; k < 5; k++) {
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
                    vector3_t frame_vel = robot_model_->GetFrameState(frame, q_rand, v_rand, pinocchio::WORLD).vel.linear();
                    for (int i = 0; i < robot_model_->GetVelDim(); i++) {
                        PerturbConfiguration(q_rand, FD_DELTA, i);
                        vector3_t frame_pos_pert = robot_model_->GetFrameState(frame, q_rand, v_rand, pinocchio::WORLD).vel.linear();
                        q_rand = q_original;

                        for (int j = 0; j < frame_vel.size(); j++) {
                            frame_fd(j, i) = (frame_pos_pert(j) - frame_vel(j))/FD_DELTA;
                            CHECK_THAT(frame_fd(j,i) - jacobian(j, i),
                                Catch::Matchers::WithinAbs(0, FD_MARGIN));
                        }
                    }

                    // ------- Velocity ------- //
                    // Analytic solution
                    jacobian.setZero();
                    HolonomicLinearizationv(0, frame, jacobian);

                    // Finite difference
                    frame_fd.setZero();
                    frame_vel = robot_model_->GetFrameState(frame, q_rand, v_rand, pinocchio::WORLD).vel.linear();
                    for (int i = 0; i < robot_model_->GetVelDim(); i++) {
                        v_rand(i) += FD_DELTA;
                        vector3_t frame_pos_pert = robot_model_->GetFrameState(frame, q_rand, v_rand, pinocchio::WORLD).vel.linear();
                        v_rand(i) -= FD_DELTA;

                        for (int j = 0; j < frame_vel.size(); j++) {
                            frame_fd(j, i) = (frame_pos_pert(j) - frame_vel(j))/FD_DELTA;
                            CHECK_THAT(frame_fd(j,i) - jacobian(j, i),
                                Catch::Matchers::WithinAbs(0, FD_MARGIN));
                        }
                    }
                }
            }
        }

        void CheckCostFunctionDerivatives() {
            PrintTestHeader("Cost Function Derivatives");
            for (int k = 0; k < 5; k++) {
                // ----- Configuration cost ----- //
                vectorx_t d_rand = robot_model_->GetRandomVel();
                vectorx_t bar_rand = robot_model_->GetRandomConfig();
                vectorx_t target_rand = robot_model_->GetRandomConfig();

                // Analytic
                vectorx_t arg;
                // cost_ FormCostFcnArg(d_rand, bar_rand, target_rand, arg);
                // vectorx_t grad_c = config_cost_fcn_->Gradient(arg);

                // Finite difference
                // vectorx_t fd_c = config_cost_fcn_->GradientFiniteDiff(arg);

                // CHECK(grad_c.isApprox(fd_c, sqrt(FD_DELTA)));

                // ----- Velocity cost ----- //
                vectorx_t vbar_rand = robot_model_->GetRandomVel();
                vectorx_t vtarget_rand = robot_model_->GetRandomVel();

                // Analytic
                // FormCostFcnArg(d_rand, vbar_rand, vtarget_rand, arg);
                // vectorx_t grad_v = vel_cost_fcn_->Gradient(arg);

                // Finite difference
                // vectorx_t fd_v = vel_cost_fcn_->GradientFiniteDiff(arg);

                // CHECK(grad_v.isApprox(fd_v, sqrt(FD_DELTA)));
            }
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
            vectorx_t state_rand = robot_model_->GetRandomState();
            BENCHMARK("mpc compute") {
                Compute(state_rand);
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
