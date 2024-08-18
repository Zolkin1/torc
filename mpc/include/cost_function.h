//
// Created by zolkin on 8/2/24.
//

#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include <eigen_utils.h>
#include <stdexcept>
#include <Eigen/Core>

#include "autodiff_fn.h"
#include "quadratic_fn.h"
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
    using sp_matrixx_t = Eigen::SparseMatrix<double, Eigen::ColMajor, long long>;

    enum CostTypes {
        Configuration = 0,
        VelocityTracking,
        TorqueReg,
        ForceReg,
    };

    // TODO: Speed up - make the ones that have explicit formulas not use autodiff
    class CostFunction {
    public:
        explicit CostFunction(const std::string& name)
            : name_(name), configured_(false), compile_derivatives_(true) {}

        void Configure(int config_size, int vel_size, int joint_size, int input_size,
            bool compile_derivatives, const std::vector<CostTypes>& costs, const std::vector<vectorx_t>& weights) {
            config_size_ = config_size;
            vel_size_ = vel_size;
            joint_size_ = joint_size;
            input_size_ = input_size;
            force_size_ = 3; // For now assuming forces are point contacts
            compile_derivatives_ = compile_derivatives;

            if (costs.size() != weights.size()) {
                throw std::runtime_error("Cost terms and weights must be the same size!");
            }

            namespace ADCG = CppAD::cg;
            namespace AD = CppAD;

            using cg_t = ADCG::CG<double>;
            using adcg_t = CppAD::AD<cg_t>;

            weights_.resize(costs.size());
            int idx = 0;
            for (const auto& cost_term : costs) {
                cost_idxs_.insert(std::pair<CostTypes, int>(cost_term, idx));
                weights_[cost_idxs_[cost_term]] = weights[idx];
                idx++;

                if (cost_term == Configuration) {
                    cost_fcn_terms_.emplace_back(std::make_unique<torc::fn::AutodiffFn<double>>(
                        CreateDefaultCost<adcg_t>(Configuration), 2*config_size_ + vel_size_,
                        compile_derivatives_, false, name_ + "_mpc_config_cost"));
                }
                else if (cost_term == VelocityTracking) {
                    cost_fcn_terms_.emplace_back(std::make_unique<torc::fn::AutodiffFn<double>>(
                        CreateDefaultCost<adcg_t>(VelocityTracking), 3*vel_size_,
                        compile_derivatives_, false, name_ + "_mpc_vel_cost"));
                } else if (cost_term == TorqueReg) {
                    cost_fcn_terms_.emplace_back(std::make_unique<torc::fn::AutodiffFn<double>>(
                                            CreateDefaultCost<adcg_t>(TorqueReg), 3*input_size_,
                                            compile_derivatives_, false, name_ + "_mpc_torque_reg_cost"));
                } else  if (cost_term == ForceReg) {
                    cost_fcn_terms_.emplace_back(std::make_unique<torc::fn::AutodiffFn<double>>(
                                                                CreateDefaultCost<adcg_t>(ForceReg), 3*force_size_,
                                                                compile_derivatives_, false, name_ + "_mpc_force_reg_cost"));
                } else {
                    std::cerr << "Cost term is not recognized!" << std::endl;
                }
                // CppAD functions are slow to evaluate, so get the double function
                cost_fcn_terms_[cost_idxs_[cost_term]]->func_ = CreateDefaultCost<double>(cost_term);
            }

            configured_ = true;
        }

        void Linearize(const vectorx_t& reference, const vectorx_t& target, const CostTypes& type, vectorx_t& linear_term) {
            if (!configured_) {
                throw std::runtime_error("Cost function not configured yet!");
            }
            if (type == Configuration) {
                if (reference.size() != config_size_ || target.size() != config_size_) {
                    std::cerr << "reference: " << reference.transpose() << std::endl;
                    std::cerr << "target: " << target.transpose() << std::endl;
                    std::cerr << "config size: " << config_size_ << std::endl;

                    throw std::runtime_error("[Cost Function] configuration linearization reference or target has the wrong size!");
                }
                vectorx_t arg;
                FormCostFcnArg(vectorx_t::Zero(vel_size_), reference, target, arg);
                linear_term.resize(vel_size_);
                linear_term = cost_fcn_terms_[cost_idxs_[Configuration]]->Gradient(arg).head(vel_size_);
            } else if (type == VelocityTracking) {
                if (reference.size() != vel_size_ || target.size() != vel_size_) {
                    throw std::runtime_error("[Cost Function] velocity linearization reference or target has the wrong size!");
                }
                vectorx_t arg;
                FormCostFcnArg(vectorx_t::Zero(vel_size_), reference, target, arg);
                linear_term.resize(vel_size_);
                linear_term = cost_fcn_terms_[cost_idxs_[VelocityTracking]]->Gradient(arg).head(vel_size_);
            } else if (type == TorqueReg) {
                if (reference.size() != input_size_ || target.size() != input_size_) {
                    throw std::runtime_error("[Cost Function] torque linearization reference or target has the wrong size!");
                }
                vectorx_t arg;
                FormCostFcnArg(vectorx_t::Zero(input_size_), reference, target, arg);
                linear_term.resize(input_size_);
                linear_term = cost_fcn_terms_[cost_idxs_[TorqueReg]]->Gradient(arg).head(input_size_);
            } else if (type == ForceReg) {
                if (reference.size() != force_size_ || target.size() != force_size_) {
                    throw std::runtime_error("[Cost Function] force linearization reference or target has the wrong size!");
                }
                vectorx_t arg;
                FormCostFcnArg(vectorx_t::Zero(force_size_), reference, target, arg);
                linear_term.resize(force_size_);
                linear_term = cost_fcn_terms_[cost_idxs_[ForceReg]]->Gradient(arg).head(force_size_);
            } else {
                throw std::runtime_error("Provided cost type not supported yet!");
            }
        }

        void Quadraticize(const vectorx_t& reference, const vectorx_t& target, const CostTypes& type, matrixx_t& hessian_term) {
            if (!configured_) {
                throw std::runtime_error("Cost function not configured yet!");
            }
            if (type == Configuration) {
                if (reference.size() != config_size_ || target.size() != config_size_) {
                    throw std::runtime_error("[Cost Function] configuration quadratic reference or target has the wrong size!");
                }
                vectorx_t arg;
                FormCostFcnArg(vectorx_t::Zero(vel_size_), reference, target, arg);
                hessian_term.resize(vel_size_, vel_size_);

                // Make sure its PSD so use Gauss-Newton Approximation
                // vectorx_t grad = cost_fcn_terms_[cost_idxs_[Configuration]]->Gradient(arg).head(vel_size_);
                // double cost = cost_fcn_terms_[cost_idxs_[Configuration]]->Evaluate(arg);
                // if (cost < 1e-6) {
                //     cost = 1e-6;
                // }
                hessian_term = 2*GetConfigurationTrackingJacobian(arg).transpose() * GetConfigurationTrackingJacobian(arg);
            } else if (type == VelocityTracking) {
                if (reference.size() != vel_size_ || target.size() != vel_size_) {
                    throw std::runtime_error("[Cost Function] velocity quadratic reference or target has the wrong size!");
                }
                vectorx_t arg;
                FormCostFcnArg(vectorx_t::Zero(vel_size_), reference, target, arg);
                hessian_term.resize(vel_size_, vel_size_);
                hessian_term = cost_fcn_terms_[cost_idxs_[VelocityTracking]]->Hessian(arg).topLeftCorner(vel_size_, vel_size_);
            } else if (type == TorqueReg) {
                if (reference.size() != input_size_ || target.size() != input_size_) {
                    throw std::runtime_error("[Cost Function] torque quadratic reference or target has the wrong size!");
                }
                vectorx_t arg;
                FormCostFcnArg(vectorx_t::Zero(input_size_), reference, target, arg);
                hessian_term.resize(input_size_, input_size_);
                hessian_term = cost_fcn_terms_[cost_idxs_[TorqueReg]]->Hessian(arg).topLeftCorner(input_size_, input_size_);
            } else if (type == ForceReg) {
                if (reference.size() != force_size_ || target.size() != force_size_) {
                    throw std::runtime_error("[Cost Function] force quadratic reference or target has the wrong size!");
                }
                vectorx_t arg;
                FormCostFcnArg(vectorx_t::Zero(force_size_), reference, target, arg);
                hessian_term.resize(force_size_, force_size_);
                hessian_term = cost_fcn_terms_[cost_idxs_[ForceReg]]->Hessian(arg).topLeftCorner(force_size_, force_size_);
            } else {
                throw std::runtime_error("Provided cost type not supported yet!");
            }
        }

        [[nodiscard]] double GetTermCost(const vectorx_t& decision_var, const vectorx_t& reference, const vectorx_t& target, const CostTypes& type) {
            if (!configured_) {
                throw std::runtime_error("Cost function not configured yet!");
            }

            vectorx_t arg;
            FormCostFcnArg(decision_var, reference, target, arg);
            return cost_fcn_terms_[cost_idxs_[type]]->Evaluate(arg);
        }


    protected:
        static constexpr int POS_VARS = 3;
        static constexpr int QUAT_VARS = 4;
        static constexpr int FLOATING_VEL = 6;
        static constexpr int FLOATING_BASE = 7;

        // TODO: Consider moving the torque and velocity constraints out
        template<class ScalarT>
        std::function<ScalarT(Eigen::VectorX<ScalarT>)> CreateDefaultCost(const CostTypes& type) {
            namespace ADCG = CppAD::cg;
            namespace AD = CppAD;

            using cg_t = ADCG::CG<double>;
            using adcg_t = CppAD::AD<cg_t>;

            int config_size = config_size_;
            int vel_size = vel_size_;
            int joint_size = joint_size_;

            vectorx_t weight = weights_[cost_idxs_[type]];

            if (type == Configuration) {
                if (weight.size() != vel_size_) {
                    throw std::runtime_error("Configuration weight has wrong size!");
                }

                // TODO: Wrap in if statement
                // if (compile_derivatives_) {
                // TODO: Do this better
                // TODO: Better hand the files
                // --------------------------------------------------- //
                // ----- Jacobian needed for Gauss-Newton approx ----- //
                // --------------------------------------------------- //
                // Tape the model
                std::vector<adcg_t> x(2*config_size_ + vel_size_);
                CppAD::Independent(x);
                Eigen::VectorX<adcg_t> eigen_x = Eigen::Map<Eigen::VectorX<adcg_t> , Eigen::Unaligned>(x.data(), x.size());
                Eigen::VectorX<adcg_t> y = {GetConfigurationDiffVector(eigen_x)};
                std::vector<adcg_t> y_std = torc::utils::EigenToStdVector(y);
                AD::ADFun<cg_t> ad_fn(x, y_std);

                // generate library source code
                ADCG::ModelCSourceGen<double> c_gen(ad_fn, this->name_ + "_jacobian");
                c_gen.setCreateJacobian(true);
                c_gen.setCreateHessian(true);
                ADCG::ModelLibraryCSourceGen<double> lib_gen(c_gen);
                ADCG::DynamicModelLibraryProcessor<double> lib_processor(lib_gen);

                // compile source code into a dynamic library
                ADCG::GccCompiler<double> compiler;
                this->config_jacobian_lib_ = lib_processor.createDynamicLibrary(compiler);
                this->config_jacobian_model_ = this->config_jacobian_lib_->model(this->name_ + "_jacobian");
                // } else {
                // }
                return [this, joint_size, vel_size, config_size, weight](const Eigen::VectorX<ScalarT>& dq_qbar_qtarget) {
                    Eigen::VectorX<ScalarT> q_diff = this->GetConfigurationDiffVector(dq_qbar_qtarget);
                    return q_diff.squaredNorm();
                };
            } else if (type == VelocityTracking) {
                if (weight.size() != vel_size_) {
                    throw std::runtime_error("Velocity tracking weight has wrong size!");
                }
                return [vel_size, weight](const Eigen::VectorX<ScalarT>& dv_vbar_vtarget) {
                    Eigen::VectorX<ScalarT> v_diff = dv_vbar_vtarget.head(vel_size) + dv_vbar_vtarget.segment(vel_size, vel_size);  // Get the current velocity
                    v_diff = v_diff - dv_vbar_vtarget.tail(vel_size);    // Get the difference between the velocity and its target
                    for (int i = 0; i < weight.size(); i++) {
                        v_diff(i) = v_diff(i) * weight(i);
                    }
                    return v_diff.squaredNorm();
                };
            } else if (type == TorqueReg) {
                if (weight.size() != input_size_) {
                    throw std::runtime_error("Torque regularization weight has wrong size!");
                }
                int input_size = input_size_;
                return [input_size, weight](const Eigen::VectorX<ScalarT>& dtau_taubar_tautarget) {
                    Eigen::VectorX<ScalarT> tau_diff = dtau_taubar_tautarget.head(input_size) + dtau_taubar_tautarget.segment(input_size, input_size);  // Get the current velocity
                    tau_diff = tau_diff - dtau_taubar_tautarget.tail(input_size);    // Get the difference between the velocity and its target
                    for (int i = 0; i < weight.size(); i++) {
                        tau_diff(i) = tau_diff(i) * weight(i);
                    }
                    return tau_diff.squaredNorm();
                };
            } else  if (type == ForceReg) {
                if (weight.size() != force_size_) {
                    throw std::runtime_error("Force regularization weight has wrong size!");
                }
                int force_size = force_size_;
                // *** Note ** that this function takes a single 3 vector of forces, not all the contact forces
                return [force_size, weight](const Eigen::VectorX<ScalarT>& dforce_forcebar_forcetarget) {
                    Eigen::VectorX<ScalarT> force_diff = dforce_forcebar_forcetarget.head(force_size)
                        + dforce_forcebar_forcetarget.segment(force_size, force_size);  // Get the current velocity
                    force_diff = force_diff - dforce_forcebar_forcetarget.tail(force_size);    // Get the difference between the velocity and its target
                    for (int i = 0; i < weight.size(); i++) {
                        force_diff(i) = force_diff(i) * weight(i);
                    }
                    return force_diff.squaredNorm();
                };
            }
            throw std::runtime_error("Unsupported cost type!");
        }

        void FormCostFcnArg(const vectorx_t& delta, const vectorx_t& bar, const vectorx_t& target, vectorx_t& arg) const {
            arg.resize(delta.size() + bar.size() + target.size());
            arg << delta, bar, target;
        }

        template<class ScalarT>
        Eigen::VectorX<ScalarT> GetConfigurationDiffVector(const Eigen::VectorX<ScalarT>& dq_qbar_qtarget) {
            Eigen::VectorX<ScalarT> q_diff = Eigen::VectorX<ScalarT>::Zero(vel_size_);
            // Floating base position difference
            q_diff.template head<POS_VARS>() = dq_qbar_qtarget.template head<POS_VARS>() + dq_qbar_qtarget.segment(vel_size_, POS_VARS)
                - dq_qbar_qtarget.segment(config_size_ + vel_size_, POS_VARS); // Get the current floating base position minus target
            // Floating base orientation difference
            Eigen::Quaternion<ScalarT> qbar, q_target;
            qbar.coeffs() = dq_qbar_qtarget.template segment<QUAT_VARS>(vel_size_ + POS_VARS);
            q_target.coeffs() = dq_qbar_qtarget.template segment<QUAT_VARS>(config_size_ + vel_size_ + POS_VARS);
            // Eigen's inverse has an if statement, so we can't use it in codegen
            qbar = Eigen::Quaternion<ScalarT>(qbar.conjugate().coeffs() / qbar.squaredNorm());   // Assumes norm > 0

            q_diff.template segment<3>(POS_VARS) = pinocchio::quaternion::log3(
                qbar * q_target
                 * pinocchio::quaternion::exp3(dq_qbar_qtarget.template segment<3>(POS_VARS)));

            // Joint differences
            q_diff.segment(FLOATING_VEL, joint_size_) =
                // dq
                dq_qbar_qtarget.segment(FLOATING_VEL, joint_size_)
                // qbar
                + dq_qbar_qtarget.segment(vel_size_ + FLOATING_BASE, joint_size_)
                // qtarget
                - dq_qbar_qtarget.segment(config_size_ + vel_size_ + FLOATING_BASE, joint_size_);
            for (int i = 0; i < weights_[cost_idxs_[Configuration]].size(); i++) {
                q_diff(i) = q_diff(i) * weights_[cost_idxs_[Configuration]](i);
            }

            return q_diff;
        }

        matrixx_t GetConfigurationTrackingJacobian(const vectorx_t& arg) {
            const std::vector<double> arg_std(arg.data(), arg.data() + arg.size());
            std::vector<double> jac = config_jacobian_model_->Jacobian(arg_std);
            // matrixx_t jac_eig = Eigen::Map<matrixx_t>(jac.data(), vel_size_, 2*config_size_ + vel_size_);
            matrixx_t jac_eig = Eigen::Map<matrixx_t>(jac.data(), 2*config_size_ + vel_size_, vel_size_);
            return jac_eig.transpose().leftCols(vel_size_);
        }

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
        std::vector<std::unique_ptr<fn::AutodiffFn<double>>> cost_fcn_terms_;
        std::vector<vectorx_t> weights_;
        std::map<CostTypes, int> cost_idxs_;

        std::unique_ptr<CppAD::cg::DynamicLib<double>> config_jacobian_lib_;
        std::unique_ptr<CppAD::cg::GenericModel<double>> config_jacobian_model_;
    private:
    };
}    // namespace torc::mpc


#endif //COST_FUNCTION_H
