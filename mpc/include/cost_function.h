//
// Created by zolkin on 8/2/24.
//

#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include <stdexcept>
#include <Eigen/Core>

#include "autodiff_fn.h"
#include "pinocchio/spatial/explog-quaternion.hpp"
#include "trajectory.h"

namespace torc::mpc {
    using vectorx_t = Eigen::VectorXd;
    using quat_t = Eigen::Quaterniond;
    using matrixx_t = Eigen::MatrixXd;
    using matrix3x_t = Eigen::Matrix3Xd;
    using matrix3_t = Eigen::Matrix3d;
    using matrix43_t = Eigen::Matrix<double, 4, 3>;
    using matrix6x_t = Eigen::Matrix<double, 6, Eigen::Dynamic>;
    using sp_matrixx_t = Eigen::SparseMatrix<double>;

    enum CostTypes {
        Configuration = 0,
        Velocity
    };

    class CostFunction {
    public:
        CostFunction()
            : configured_(false) {}

        void Configure(int config_size, int vel_size, int joint_size, const std::vector<CostTypes>& costs, const std::vector<vectorx_t>& weights) {
            config_size_ = config_size;
            vel_size_ = vel_size;
            joint_size_ = joint_size;

            namespace ADCG = CppAD::cg;
            namespace AD = CppAD;

            using cg_t = ADCG::CG<double>;
            using adcg_t = CppAD::AD<cg_t>;

            if (costs.size() != weights.size()) {
                throw std::runtime_error("Cost terms and weights must be the same size!");
            }

            weights_.resize(costs.size());
            int idx = 0;
            for (const auto& cost_term : costs) {
                cost_idxs_.insert(std::pair<CostTypes, int>(cost_term, idx));
                weights_[cost_idxs_[cost_term]] = weights[idx];
                idx++;


                if (cost_term == Configuration) {
                    cost_fcn_terms_.emplace_back(std::make_unique<torc::fn::AutodiffFn<double>>(
                        CreateDefaultCost<adcg_t>(Configuration), 2*config_size_ + vel_size_, true, false, "mpc_config_cost"));
                }
                else if (cost_term == Velocity) {
                    cost_fcn_terms_.emplace_back(std::make_unique<torc::fn::AutodiffFn<double>>(
                        CreateDefaultCost<adcg_t>(Velocity), 3*vel_size_, true, false, "mpc_vel_cost"));
                }
                // CppAD functions are slow to evaluate, so get the double function
                cost_fcn_terms_[cost_idxs_[cost_term]]->func_ = CreateDefaultCost<double>(cost_term);
            }

            configured_ = true;
        }

        void Linearize(const Trajectory& traj, const std::vector<vectorx_t>& q_target,
            const std::vector<vectorx_t>& v_target, vectorx_t& linear_term) {
            // TODO: Implement
        }

        void Quadraticize(const Trajectory& traj, const std::vector<vectorx_t>& q_target,
            const std::vector<vectorx_t>& v_target, matrixx_t& hessian_term) {
            // TODO: Implement
        }

        double GetCost(const vectorx_t& qp_res) {
            // TODO: Implement
        }

        double GetCost(const Trajectory& traj) {
            // TODO: Implement
        }

        [[nodiscard]] double GetTermCost(const vectorx_t& decision_var, const vectorx_t& reference, const vectorx_t& target, const CostTypes& type) {
            vectorx_t arg;
            FormCostFcnArg(decision_var, reference, target, arg);
            return cost_fcn_terms_[cost_idxs_[type]]->Evaluate(arg);
        }


    protected:
        static constexpr int POS_VARS = 3;
        static constexpr int QUAT_VARS = 4;
        static constexpr int FLOATING_VEL = 6;
        static constexpr int FLOATING_BASE = 7;

        template<class ScalarT>
        std::function<ScalarT(Eigen::VectorX<ScalarT>)> CreateDefaultCost(const CostTypes& type) {
            int config_size = config_size_;
            int vel_size = vel_size_;
            int joint_size = joint_size_;

            vectorx_t weight = weights_[cost_idxs_[type]];

            if (type == Configuration) {
                if (weight.size() != config_size_) {
                    throw std::runtime_error("Configuration weight has wrong size!");
                }
                return [config_size, vel_size, joint_size, weight](const Eigen::VectorX<ScalarT>& dq_qbar_qtarget) {
                    Eigen::VectorX<ScalarT> q_diff = Eigen::VectorX<ScalarT>::Zero(vel_size);
                    // Floating base position difference
                    q_diff.template head<POS_VARS>() = dq_qbar_qtarget.template head<POS_VARS>() + dq_qbar_qtarget.segment(vel_size, POS_VARS)
                        - dq_qbar_qtarget.segment(config_size + vel_size, POS_VARS); // Get the current floating base position minus target

                    // Floating base orientation difference
                    Eigen::Quaternion<ScalarT> qbar, q_target;
                    qbar.coeffs() = dq_qbar_qtarget.template segment<QUAT_VARS>(config_size + vel_size + POS_VARS);
                    q_target.coeffs() = dq_qbar_qtarget.template segment<QUAT_VARS>(vel_size + POS_VARS);
                    // Eigen's inverse has an if statement, so we can't use it in codegen
                    qbar = Eigen::Quaternion<ScalarT>(qbar.conjugate().coeffs() / qbar.squaredNorm());   // Assumes norm > 0

                    q_diff.template segment<3>(POS_VARS) = pinocchio::quaternion::log3(
                        qbar * q_target
                         * pinocchio::quaternion::exp3(dq_qbar_qtarget.template segment<3>(POS_VARS)));

                    // Joint differences
                    q_diff.segment(FLOATING_VEL, joint_size) =
                        // dq
                        dq_qbar_qtarget.segment(FLOATING_VEL, joint_size)
                        // qbar
                        + dq_qbar_qtarget.segment(vel_size + FLOATING_BASE, joint_size)
                        // qtarget
                        - dq_qbar_qtarget.segment(config_size + vel_size + FLOATING_BASE, joint_size);
                    for (int i = 0; i < weight.size(); i++) {
                        q_diff(i) = q_diff(i) * weight(i);
                    }
                    return q_diff.squaredNorm();
                };
            } else if (type == Velocity) {
                if (weight.size() != vel_size_) {
                    throw std::runtime_error("Velocity weight has wrong size!");
                }
                return [vel_size, weight](const Eigen::VectorX<ScalarT>& dv_vbar_vtarget) {
                    Eigen::VectorX<ScalarT> v_diff = dv_vbar_vtarget.head(vel_size) + dv_vbar_vtarget.segment(vel_size, vel_size);  // Get the current velocity
                    v_diff = v_diff - dv_vbar_vtarget.tail(vel_size);    // Get the difference between the velocity and its target
                    for (int i = 0; i < weight.size(); i++) {
                        v_diff(i) = v_diff(i) * weight(i);
                    }
                    return v_diff.squaredNorm();
                };
            }
            throw std::runtime_error("Unsupported cost type!");
        }

        void FormCostFcnArg(const vectorx_t& delta, const vectorx_t& bar, const vectorx_t& target, vectorx_t& arg) const {
            arg.resize(delta.size() + bar.size() + target.size());
            arg << delta, bar, target;
        }

        int config_size_;
        int vel_size_;
        int joint_size_;

        std::vector<std::unique_ptr<fn::ExplicitFn<double>>> cost_fcn_terms_;
        std::vector<vectorx_t> weights_;
        std::map<CostTypes, int> cost_idxs_;
        // std::unique_ptr<fn::ExplicitFn<double>> config_cost_fcn_;
        // std::unique_ptr<fn::ExplicitFn<double>> vel_cost_fcn_;

        bool configured_;
    private:
    };
}    // namespace torc::mpc


#endif //COST_FUNCTION_H
