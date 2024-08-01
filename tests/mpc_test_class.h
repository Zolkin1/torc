//
// Created by zolkin on 7/31/24.
//

#ifndef MPC_TEST_CLASS_H
#define MPC_TEST_CLASS_H

#include <catch2/catch_test_macros.hpp>

#include "full_order_mpc.h"

namespace torc::mpc {
    class MpcTestClass : public FullOrderMpc {
    public:
        MpcTestClass(const fs::path& config_file, const fs::path& model_path)
            : FullOrderMpc(config_file, model_path) {}

        void CheckQuaternionIntegration() {
            // Make some
        }

        void CheckSwingHeightLin() {
            PrintTestHeader("Swing Height Linearization");
            constexpr double FD_MARGIN = 1e-5;

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
            constexpr double FD_MARGIN = 1e-5;

            for (int k = 0; k < 1; k++) {
                vectorx_t q_rand = robot_model_->GetRandomConfig();
                vectorx_t v_rand = robot_model_->GetRandomVel();
                vectorx_t q_original = q_rand;
                traj_.SetConfiguration(0, q_rand);
                traj_.SetVelocity(0, v_rand);

                for (const auto& frame : contact_frames_) {
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
                }
            }
        }

        void BenchmarkQuaternionIntegrationLin() {
            BENCHMARK("quaternion integration lin") {
                auto deriv = QuatIntegrationLinearizationXi(0);
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

        void BenchmarkCompute() {
            vectorx_t state_rand = robot_model_->GetRandomState();
            BENCHMARK("mpc compute") {
                Compute(state_rand);
            };
        }
    protected:
    private:
        void PrintTestHeader(const std::string& name) {
            using std::setw;
            using std::setfill;

            const int total_width = 50;
            std::cout << setfill('=') << setw(total_width/2 - name.size()/2) << " " << name << " " << setw(total_width/2 - name.size()/2) << "" << std::endl;
        }

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
    };
} // namespacre torc::mpc

#endif //MPC_TEST_CLASS_H
