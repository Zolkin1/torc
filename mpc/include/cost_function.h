//
// Created by zolkin on 8/2/24.
//

#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include <eigen_utils.h>
#include <stdexcept>
#include <Eigen/Core>
#include <filesystem>
#include <pinocchio/algorithm/joint-configuration.hpp>

//#include "autodiff_fn.h"
//#include "quadratic_fn.h"
#include "pinocchio/codegen/cppadcg.hpp"
#include "pinocchio/spatial/explog-quaternion.hpp"
#include "trajectory.h"
#include "cpp_ad_interface.h"
#include "full_order_rigid_body.h"

namespace torc::mpc {
    using vectorx_t = Eigen::VectorXd;
    using quat_t = Eigen::Quaterniond;
    using matrixx_t = Eigen::MatrixXd;
    using matrix3x_t = Eigen::Matrix3Xd;
    using matrix3_t = Eigen::Matrix3d;
    using matrix43_t = Eigen::Matrix<double, 4, 3>;
    using matrix6x_t = Eigen::Matrix<double, 6, Eigen::Dynamic>;
    using sp_matrixx_t = Eigen::SparseMatrix<double, Eigen::ColMajor, long long>;

    enum CostTypes {
        Configuration = 0,
        VelocityTracking,
        TorqueReg,
        ForceReg,
    };

    class CostFunction {
    public:
        explicit CostFunction(const std::string& name)
            : name_(name), configured_(false), compile_derivatives_(true) {}

        void Configure(const std::unique_ptr<torc::models::FullOrderRigidBody>& model,
            bool compile_derivatives, const std::vector<CostTypes>& costs, const std::vector<vectorx_t>& weights,
            std::filesystem::path deriv_libs_path) {

            ad_pin_model_ = model->GetADPinModel();
            ad_pin_data_ = model->GetADPinData();

            config_size_ = model->GetConfigDim();
            vel_size_ = model->GetVelDim();
            joint_size_ = model->GetNumInputs();
            input_size_ = model->GetNumInputs();
            force_size_ = 3; // For now assuming forces are point contacts

            compile_derivatives_ = compile_derivatives;

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
                    if (weights_[cost_idxs_[cost_term]].size() != vel_size_) {
                        throw std::runtime_error("Configuration tracking weight has wrong size!");
                    }

                    cost_fcn_terms_.emplace_back(std::make_unique<torc::ad::CppADInterface>(
                            std::bind(&CostFunction::ConfigurationTrackingCost, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                            name_ + "_config_tracking_cost",
                            deriv_libs_path,
                            torc::ad::DerivativeOrder::FirstOrder, vel_size_, 2*config_size_ + vel_size_,
                            compile_derivatives_));
                }
                else if (cost_term == VelocityTracking) {
                    if (weights_[cost_idxs_[cost_term]].size() != vel_size_) {
                        throw std::runtime_error("Velocity tracking weight has wrong size!");
                    }

                    cost_fcn_terms_.emplace_back(std::make_unique<torc::ad::CppADInterface>(
                            std::bind(&CostFunction::VelocityTrackingCost, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                            name_ + "_mpc_vel_cost",
                            deriv_libs_path,
                            torc::ad::DerivativeOrder::SecondOrder, vel_size_, 3*vel_size_,
                            compile_derivatives_));
                } else if (cost_term == TorqueReg) {
                    if (weights_[cost_idxs_[cost_term]].size() != input_size_) {
                        throw std::runtime_error("Torque regularization weight has wrong size!");
                    }

                    cost_fcn_terms_.emplace_back(std::make_unique<torc::ad::CppADInterface>(
                            std::bind(&CostFunction::TorqueTrackingCost, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                            name_ + "_mpc_torque_reg_cost",
                            deriv_libs_path,
                            torc::ad::DerivativeOrder::SecondOrder, input_size_, 3*input_size_,
                            compile_derivatives_));
                } else  if (cost_term == ForceReg) {
                    if (weights_[cost_idxs_[cost_term]].size() != force_size_) {
                        throw std::runtime_error("Force regularization weight has wrong size!");
                    }

                    cost_fcn_terms_.emplace_back(std::make_unique<torc::ad::CppADInterface>(
                            std::bind(&CostFunction::ForceTrackingCost, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                            name_ + "_mpc_force_reg_cost",
                            deriv_libs_path,
                            torc::ad::DerivativeOrder::SecondOrder, force_size_, 3*force_size_,
                            compile_derivatives_));
                } else {
                    std::cerr << "Cost term is not recognized!" << std::endl;
                }
                // CppAD functions are slow to evaluate, so get the double function
//                cost_fcn_terms_[cost_idxs_[cost_term]]->func_ = CreateDefaultCost<double>(cost_term);
            }

            configured_ = true;
        }

        void GetApproximation(const vectorx_t& reference, const vectorx_t& target, vectorx_t& linear_term,
                              matrixx_t& hessian_term, const CostTypes& type) {
            vectorx_t p(reference.size() + target.size() + weights_[cost_idxs_[type]].size());
            p << reference, target, weights_[cost_idxs_[type]];

            if (type == Configuration) {
                if (reference.size() != config_size_ || target.size() != config_size_) {
                    throw std::runtime_error("[Cost Function] configuration approx reference or target has the wrong size!");
                }

                matrixx_t jac;
                cost_fcn_terms_[cost_idxs_[type]]->GetGaussNewton(vectorx_t::Zero(vel_size_), p, jac, hessian_term);
                ConvertJacobianToVectorSum(jac, linear_term);
                hessian_term = 2*hessian_term;
            } else if (type == VelocityTracking) {
                if (reference.size() != vel_size_ || target.size() != vel_size_) {
                    throw std::runtime_error("[Cost Function] velocity approx reference or target has the wrong size!");
                }

                matrixx_t jac;
                cost_fcn_terms_[cost_idxs_[type]]->GetJacobian(vectorx_t::Zero(vel_size_), p, jac);
                ConvertJacobianToVectorSum(jac, linear_term);

                vectorx_t w = vectorx_t::Constant(vel_size_, 1);
                cost_fcn_terms_[cost_idxs_[type]]->GetHessian(vectorx_t::Zero(vel_size_), p, w, hessian_term);
            } else if (type == TorqueReg) {
                if (reference.size() != input_size_ || target.size() != input_size_) {
                    throw std::runtime_error("[Cost Function] torque approx reference or target has the wrong size!");
                }

                matrixx_t jac;
                cost_fcn_terms_[cost_idxs_[type]]->GetJacobian(vectorx_t::Zero(input_size_), p, jac);
                ConvertJacobianToVectorSum(jac, linear_term);

                vectorx_t w = vectorx_t::Constant(input_size_, 1);
                cost_fcn_terms_[cost_idxs_[type]]->GetHessian(vectorx_t::Zero(input_size_), p, w, hessian_term);
            } else if (type == ForceReg) {
                if (reference.size() != force_size_ || target.size() != force_size_) {
                    throw std::runtime_error("[Cost Function] force approx reference or target has the wrong size!");
                }

                matrixx_t jac;
                cost_fcn_terms_[cost_idxs_[type]]->GetJacobian(vectorx_t::Zero(force_size_), p, jac);
                ConvertJacobianToVectorSum(jac, linear_term);

                vectorx_t w = vectorx_t::Constant(force_size_, 1);
                cost_fcn_terms_[cost_idxs_[type]]->GetHessian(vectorx_t::Zero(force_size_), p, w, hessian_term);
            } else {
                throw std::runtime_error("Provided cost type not supported yet!");
            }
        }

        torc::ad::sparsity_pattern_t GetJacobianSparsityPattern(const CostTypes& type) {
            return cost_fcn_terms_[cost_idxs_[type]]->GetJacobianSparsityPatternSet();
        }

        torc::ad::sparsity_pattern_t GetGaussNewtonSparsityPattern(const CostTypes& type) {
            return cost_fcn_terms_[cost_idxs_[type]]->GetGaussNewtonSparsityPatternSet();
        }

        torc::ad::sparsity_pattern_t GetHessianSparsityPattern(const CostTypes& type) {
            return cost_fcn_terms_[cost_idxs_[type]]->GetHessianSparsityPatternSet();
        }

        [[nodiscard]] double GetTermCost(const vectorx_t& decision_var, const vectorx_t& reference, const vectorx_t& target, const CostTypes& type) {
            if (!configured_) {
                throw std::runtime_error("Cost function not configured yet!");
            }

            vectorx_t p(reference.size() + target.size() + weights_[cost_idxs_[type]].size());
            p << reference, target, weights_[cost_idxs_[type]];
            vectorx_t y = vectorx_t::Zero(cost_fcn_terms_[cost_idxs_[type]]->GetRangeSize());
            cost_fcn_terms_[cost_idxs_[type]]->GetFunctionValue(decision_var, p, y);
            return y.sum();     // Assumes all the functions have the form of a norm
        }


    protected:
        static void ConvertJacobianToVectorSum(matrixx_t& jac, vectorx_t& linear_term) {
            jac.transposeInPlace();
            linear_term.resize(jac.rows());
            linear_term.setZero();

            for (int col = 0; col < jac.cols(); col++) {
                linear_term += jac.col(col);
            }
        }

        static constexpr int POS_VARS = 3;
        static constexpr int QUAT_VARS = 4;
        static constexpr int FLOATING_VEL = 6;
        static constexpr int FLOATING_BASE = 7;

        // ------------------------------------ //
        // ---------- Cost Functions ---------- //
        // ------------------------------------ //
        void ConfigurationTrackingCost(const torc::ad::ad_vector_t& dq,
                                        const torc::ad::ad_vector_t& qref_qtarget_weight,
                                        torc::ad::ad_vector_t& q_diff) const {
            // I'd like to just call pinocchio's integrate here, but that will require a templated model, which I currently don't have

            q_diff.setZero(vel_size_);

            // Floating base position difference
            q_diff.head<POS_VARS>() = dq.head<POS_VARS>() + qref_qtarget_weight.head<POS_VARS>()
                                               - qref_qtarget_weight.segment(config_size_, POS_VARS); // Get the current floating base position minus target
            // Floating base orientation difference
            Eigen::Quaternion<torc::ad::adcg_t> qbar, q_target;
            qbar.coeffs() = qref_qtarget_weight.segment<QUAT_VARS>(POS_VARS);

            q_target.coeffs() = qref_qtarget_weight.segment<QUAT_VARS>(config_size_ + POS_VARS);

            // Eigen's inverse has an if statement, so we can't use it in codegen
            q_target = Eigen::Quaternion<torc::ad::adcg_t>(q_target.conjugate().coeffs() / q_target.squaredNorm());   // Assumes norm > 0

            q_diff.segment<3>(POS_VARS) = pinocchio::quaternion::log3(
                    q_target * qbar
                    * pinocchio::quaternion::exp3(dq.segment<3>(POS_VARS)));

            // Joint differences
            q_diff.segment(FLOATING_VEL, joint_size_) =
                    // dq
                    dq.segment(FLOATING_VEL, joint_size_)
                    // qbar
                    + qref_qtarget_weight.segment(FLOATING_BASE, joint_size_)
                    // qtarget
                    - qref_qtarget_weight.segment(config_size_ + FLOATING_BASE, joint_size_);

            for (int i = 0; i < vel_size_; i++) {
                q_diff(i) = CppAD::pow(q_diff(i) * qref_qtarget_weight(2*config_size_ + i), 2);
            }
        }

        void VelocityTrackingCost(const torc::ad::ad_vector_t& dv,
                                    const torc::ad::ad_vector_t& vref_vtarget_weight,
                                    torc::ad::ad_vector_t& v_diff) const {
            // Get the current velocity
            v_diff = dv + vref_vtarget_weight.head(vel_size_);

            // Get the difference between the velocity and its target
            v_diff = v_diff - vref_vtarget_weight.segment(vel_size_, vel_size_);

            // Multiply by the weights
            for (int i = 0; i < vel_size_; i++) {
                v_diff(i) = CppAD::pow(v_diff(i) * vref_vtarget_weight(2*vel_size_ + i), 2);
            }
        }

        void TorqueTrackingCost(const torc::ad::ad_vector_t& dtau,
                                   const torc::ad::ad_vector_t& tauref_tautarget_weight,
                                   torc::ad::ad_vector_t& tau_diff) const {
            // Get the current torque
            tau_diff = dtau + tauref_tautarget_weight.head(input_size_);

            // Get the difference between the velocity and its target
            tau_diff = tau_diff - tauref_tautarget_weight.segment(input_size_, input_size_);

            // Multiply by the weights
            for (int i = 0; i < input_size_; i++) {
                tau_diff(i) = CppAD::pow(tau_diff(i) * tauref_tautarget_weight(2*input_size_ + i), 2);
            }
        }

        void ForceTrackingCost(const torc::ad::ad_vector_t& df,
                                 const torc::ad::ad_vector_t& fref_ftarget_weight,
                                 torc::ad::ad_vector_t& force_diff) const {
            // Get the current torque
            force_diff = df + fref_ftarget_weight.head(force_size_);

            // Get the difference between the velocity and its target
            force_diff = force_diff - fref_ftarget_weight.segment(force_size_, force_size_);

            // Multiply by the weights
            for (int i = 0; i < force_size_; i++) {
                force_diff(i) = CppAD::pow(force_diff(i) * fref_ftarget_weight(2 * force_size_ + i), 2);
            }
        }

        /**
         * @brief Calculates the error in the foot location relative to the Raibert heuristic
         * @param dq_dv
         * @param q_v_ts_z0
         * @param foot_error
         */
//        void RaibertCost(const torc::ad::ad_vector_t& dq_dv,        // Change in configuration and velocity
//                         const torc::ad::ad_vector_t& q_v_ts_z0,    // configuration, velocity, stance time, nominal height
//                         torc::ad::ad_vector_t& foot_error) {       // Error on the foot location
//            const torc::ad::ad_vector_t& dq = dq_dv.head(vel_size_);
//            const torc::ad::ad_vector_t& dv = dq_dv.tail(vel_size_);
//            const torc::ad::ad_vector_t& q = q_v_ts_z0.head(config_size_);
//            const torc::ad::ad_vector_t& v = q_v_ts_z0.segment(config_size_, vel_size_);
//            const torc::ad::adcg_t& ts = q_v_ts_z0(config_size_ + vel_size_);
//            const torc::ad::adcg_t& z0 = q_v_ts_z0(config_size_ + vel_size_ + 1);
//
//            // TODO: Check this
//            // ----- Determine the current foot position ----- //
//            // Get the current configuration
//            torc::ad::ad_vector_t q_new = pinocchio::integrate(ad_pin_model_, q, dq);
//            torc::ad::ad_vector_t
//
//        }

        // ----------------------------------- //
        // --------- Member Variables -------- //
        // ----------------------------------- //
        std::string name_;
        bool configured_;
        bool compile_derivatives_;

        int config_size_{};
        int vel_size_{};
        int joint_size_{};
        int input_size_{};
        int force_size_{};
        int nodes_{};

        // Using explicit function here seemed to cause memory leaks
        // Also unclear if it is the memory leaks or the type here, but the MPC is notably quicker now
        std::vector<std::unique_ptr<torc::ad::CppADInterface>> cost_fcn_terms_;
        std::vector<vectorx_t> weights_;
        std::map<CostTypes, int> cost_idxs_;

        std::unique_ptr<CppAD::cg::DynamicLib<double>> config_jacobian_lib_;
        std::unique_ptr<CppAD::cg::GenericModel<double>> config_jacobian_model_;

        torc::models::ad_pin_model_t ad_pin_model_;
        std::shared_ptr<torc::models::ad_pin_data_t> ad_pin_data_;
    private:
    };
}    // namespace torc::mpc


#endif //COST_FUNCTION_H
