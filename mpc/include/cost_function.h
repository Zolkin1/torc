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
#include <pinocchio/algorithm/frames.hpp>

//#include "autodiff_fn.h"
//#include "quadratic_fn.h"
#include "pinocchio/codegen/cppadcg.hpp"
#include "pinocchio/spatial/explog-quaternion.hpp"
#include "trajectory.h"
#include "cpp_ad_interface.h"
#include "full_order_rigid_body.h"
#include "pinocchio_interface.h"

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
        ForwardKinematics,
        FootPolytope,
        CentroidalConfiguration,
        CentroidalVelocity,
        CentroidalForce,
    };

    /**
     * A struct to hold all of the constants associated with costs.
     * Only the fields that the constraint uses need to be filled, the others can be
     * left empty.
     *
     * Constraint name, weight, and type MUST always be populated.
     */
    struct CostData {
        CostTypes type;
        vectorx_t weight;
        std::string frame_name;
        std::string constraint_name;        // TODO: Change to cost name!
        ad::sparsity_pattern_t sp_pattern;
        // For cost relaxation
        double delta;
        double mu;
    };

    class CostFunction {
    public:
        explicit CostFunction(const std::string& name)
            : name_(name), configured_(false), compile_derivatives_(true) {}

        void Configure(const std::shared_ptr<torc::models::FullOrderRigidBody>& model,
            bool compile_derivatives, std::vector<CostData> cost_data,
            std::filesystem::path deriv_libs_path) {

            cost_data_ = std::move(cost_data);

            ad_pin_model_ = model->GetADPinModel();
            ad_pin_data_ = model->GetADPinData();

            config_size_ = model->GetConfigDim();
            vel_size_ = model->GetVelDim();
            joint_size_ = model->GetNumInputs();
            input_size_ = model->GetNumInputs();
            force_size_ = 3; // For now assuming forces are point contacts

            compile_derivatives_ = compile_derivatives;

            int idx = 0;
            for (const auto& data : cost_data_) {
                idx++;

                if (data.type == Configuration) {
                    if (data.weight.size() != vel_size_) {
                        throw std::runtime_error("Configuration tracking weight has wrong size!");
                    }

                    cost_fcn_terms_.emplace(data.constraint_name, std::make_unique<torc::ad::CppADInterface>(
                            std::bind(&CostFunction::ConfigurationTrackingCost, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                            name_ + data.constraint_name + "_config_tracking_cost",
                            deriv_libs_path,
                            torc::ad::DerivativeOrder::FirstOrder, vel_size_, 2*config_size_ + vel_size_,
                            compile_derivatives_));
                }
                else if (data.type == VelocityTracking) {
                    if (data.weight.size() != vel_size_) {
                        throw std::runtime_error("Velocity tracking weight has wrong size!");
                    }

                    cost_fcn_terms_.emplace(data.constraint_name, std::make_unique<torc::ad::CppADInterface>(
                            std::bind(&CostFunction::VelocityTrackingCost, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                            name_ + data.constraint_name + "_mpc_vel_cost",
                            deriv_libs_path,
                            torc::ad::DerivativeOrder::SecondOrder, vel_size_, 3*vel_size_,
                            compile_derivatives_));
                } else if (data.type == TorqueReg) {
                    if (data.weight.size()!= input_size_) {
                        throw std::runtime_error("Torque regularization weight has wrong size!");
                    }

                    cost_fcn_terms_.emplace(data.constraint_name, std::make_unique<torc::ad::CppADInterface>(
                            std::bind(&CostFunction::TorqueTrackingCost, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                            name_ + data.constraint_name + "_mpc_torque_reg_cost",
                            deriv_libs_path,
                            torc::ad::DerivativeOrder::SecondOrder, input_size_, 3*input_size_,
                            compile_derivatives_));
                } else  if (data.type == ForceReg) {
                    if (data.weight.size() != force_size_) {
                        throw std::runtime_error("Force regularization weight has wrong size!");
                    }

                    cost_fcn_terms_.emplace(data.constraint_name, std::make_unique<torc::ad::CppADInterface>(
                            std::bind(&CostFunction::ForceTrackingCost, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                            name_ + data.constraint_name + "_mpc_force_reg_cost",
                            deriv_libs_path,
                            torc::ad::DerivativeOrder::SecondOrder, force_size_, 3*force_size_,
                            compile_derivatives_));
                } else if (data.type == ForwardKinematics) {
                    if (data.weight.size() != POS_VARS) {
                        throw std::runtime_error("Force regularization weight has wrong size!");
                    }

                    cost_fcn_terms_.emplace(data.constraint_name, std::make_unique<torc::ad::CppADInterface>(
                            std::bind(&CostFunction::FkCost, this, model->GetFrameIdx(data.frame_name), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                            name_ + data.constraint_name + "_mpc_fk_cost",
                            deriv_libs_path,
                            torc::ad::DerivativeOrder::FirstOrder, vel_size_, config_size_ + 2*POS_VARS,
                            compile_derivatives_));
                } else if (data.type == FootPolytope) {
                    if (data.weight.size() != POLYTOPE_SIZE) {
                        throw std::runtime_error("Foot polytope weight has wrong size!");
                    }

                    cost_fcn_terms_.emplace(data.constraint_name, std::make_unique<torc::ad::CppADInterface>(
                            std::bind(&CostFunction::FootPolytopeCost, this, model->GetFrameIdx(data.frame_name), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                            name_ + data.constraint_name + "_mpc_polytope_cost",
                            deriv_libs_path,
                            torc::ad::DerivativeOrder::FirstOrder, vel_size_, config_size_ + POLYTOPE_SIZE + POLYTOPE_SIZE + 2 + POLYTOPE_SIZE,
                            true)); //compile_derivatives_));
                } else {
                    std::cerr << "Cost term is not recognized!" << std::endl;
                }
            }

            configured_ = true;
        }

        /**
         * @brief returns the approximation of the function at x = 0
         * @param reference
         * @param target
         * @param linear_term
         * @param hessian_term
         * @param type
         */
        void GetApproximation(const vectorx_t& reference, const vectorx_t& target, vectorx_t& linear_term,
                              matrixx_t& hessian_term, const std::string& name) {
            CostData* data;
            for (auto & i : cost_data_) {
                if (i.constraint_name == name) {
                    data = &i;
                    break;
                }
            }

            if (data == nullptr) {
                throw std::runtime_error("Cost name not found!");
            }

            vectorx_t p(reference.size() + target.size() + data->weight.size());
            p << reference, target, data->weight;

            if (data->type == Configuration) {
                if (reference.size() != config_size_ || target.size() != config_size_) {
                    throw std::runtime_error("[Cost Function] configuration approx reference or target has the wrong size!");
                }

                matrixx_t jac;
                cost_fcn_terms_[name]->GetGaussNewton(vectorx_t::Zero(vel_size_), p, jac, hessian_term);
                hessian_term = 2*hessian_term;

                vectorx_t y;
                cost_fcn_terms_[name]->GetFunctionValue(vectorx_t::Zero(vel_size_), p, y);
                linear_term = 2*jac.transpose()*y;

            } else if (data->type == VelocityTracking) {
                if (reference.size() != vel_size_ || target.size() != vel_size_) {
                    throw std::runtime_error("[Cost Function] velocity approx reference or target has the wrong size!");
                }

                matrixx_t jac;
                cost_fcn_terms_[name]->GetJacobian(vectorx_t::Zero(vel_size_), p, jac);

                vectorx_t y;
                cost_fcn_terms_[name]->GetFunctionValue(vectorx_t::Zero(vel_size_), p, y);
                linear_term = 2*jac.transpose()*y;

                cost_fcn_terms_[name]->GetHessian(vectorx_t::Zero(vel_size_), p, 2*y, hessian_term);
                hessian_term += 2*jac.transpose() * jac;

            } else if (data->type == TorqueReg) {
                if (reference.size() != input_size_ || target.size() != input_size_) {
                    throw std::runtime_error("[Cost Function] torque approx reference or target has the wrong size!");
                }

                matrixx_t jac;
                cost_fcn_terms_[name]->GetJacobian(vectorx_t::Zero(input_size_), p, jac);

                vectorx_t y;
                cost_fcn_terms_[name]->GetFunctionValue(vectorx_t::Zero(input_size_), p, y);
                linear_term = 2*jac.transpose()*y;

                cost_fcn_terms_[name]->GetHessian(vectorx_t::Zero(input_size_), p, 2*y, hessian_term);
                hessian_term += 2*jac.transpose() * jac;
            } else if (data->type == ForceReg) {
                if (reference.size() != force_size_ || target.size() != force_size_) {
                    throw std::runtime_error("[Cost Function] force approx reference or target has the wrong size!");
                }

                matrixx_t jac;
                cost_fcn_terms_[name]->GetJacobian(vectorx_t::Zero(force_size_), p, jac);

                vectorx_t y;
                cost_fcn_terms_[name]->GetFunctionValue(vectorx_t::Zero(force_size_), p, y);
                linear_term = 2*jac.transpose()*y;

                cost_fcn_terms_[name]->GetHessian(vectorx_t::Zero(force_size_), p, 2*y, hessian_term);
                hessian_term += 2*jac.transpose() * jac;
            } else if (data->type == ForwardKinematics) {
                if (reference.size() != config_size_ || target.size() != POS_VARS) {
                    throw std::runtime_error("[Cost Function] fk approx reference or target has the wrong size!");
                }

                matrixx_t jac;
                cost_fcn_terms_[name]->GetGaussNewton(vectorx_t::Zero(vel_size_), p, jac, hessian_term);
                hessian_term = 2*hessian_term;

                // hessian_term = hessian_term + 1e-5*matrixx_t::Identity(hessian_term.rows(), hessian_term.cols());       // Ensures that we are PSD

                // ----- PSD Checks
                // if (!hessian_term.isApprox(hessian_term.transpose())) {
                //     throw std::runtime_error("[Cost Function] hessian_term is not symmetric!");
                // }
                // const auto ldlt = hessian_term.template selfadjointView<Eigen::Upper>().ldlt();
                // if (ldlt.info() == Eigen::NumericalIssue || !ldlt.isPositive()) {
                //     throw std::runtime_error("[Cost Function] hessiam term is not PSD!");
                // }
                //
                // std::cout << "hessian term is PSD" << std::endl;

                vectorx_t y;
                cost_fcn_terms_[name]->GetFunctionValue(vectorx_t::Zero(vel_size_), p, y);
                linear_term = 2*jac.transpose()*y;
            } else if (data->type == FootPolytope) {
                if (reference.size() != config_size_ || target.size() != 2*POLYTOPE_SIZE + 2) {
                    throw std::runtime_error("[Cost Function] polytope approx reference or target has the wrong size! Expected "
                    + std::to_string(cost_fcn_terms_[name]->GetParameterSize()) + " but got " + std::to_string(reference.size()) + "!");
                }

                matrixx_t jac;
                cost_fcn_terms_[name]->GetGaussNewton(vectorx_t::Zero(vel_size_), p, jac, hessian_term);
                hessian_term = 2*hessian_term;

                // hessian_term = hessian_term + 1e-5*matrixx_t::Identity(hessian_term.rows(), hessian_term.cols());       // Ensures that we are PSD

                // std::cout << "hess: \n" << hessian_term << std::endl;

                vectorx_t y;
                cost_fcn_terms_[name]->GetFunctionValue(vectorx_t::Zero(vel_size_), p, y);
                linear_term = 2*jac.transpose()*y;
            } else {
                throw std::runtime_error("[Cost Function] Provided cost type not supported yet!");
            }
        }

        torc::ad::sparsity_pattern_t GetJacobianSparsityPattern(const std::string& name) {
            return cost_fcn_terms_[name]->GetJacobianSparsityPatternSet();
        }

        torc::ad::sparsity_pattern_t GetGaussNewtonSparsityPattern(const std::string& name) {
            return cost_fcn_terms_[name]->GetGaussNewtonSparsityPatternSet();
        }

        torc::ad::sparsity_pattern_t GetHessianSparsityPattern(const std::string& name) {
            return ad::CppADInterface::GetUnion(cost_fcn_terms_[name]->GetHessianSparsityPatternSet(),
                cost_fcn_terms_[name]->GetGaussNewtonSparsityPatternSet());
        }

        [[nodiscard]] double GetTermCost(const vectorx_t& decision_var, const vectorx_t& reference, const vectorx_t& target, const std::string& name) const {
            if (!configured_) {
                throw std::runtime_error("Cost function not configured yet!");
            }

            CostData data;
            for (const auto & i : cost_data_) {
                if (i.constraint_name == name) {
                    data = i;
                    break;
                }
            }

            vectorx_t p(reference.size() + target.size() + data.weight.size());
            p << reference, target, data.weight;
            vectorx_t y = vectorx_t::Zero(cost_fcn_terms_.at(name)->GetRangeSize());
            cost_fcn_terms_.at(name)->GetFunctionValue(decision_var, p, y);

            // if (cost_fcn_terms_.at(name)->GetParameterSize() == config_size_ + POLYTOPE_SIZE + POLYTOPE_SIZE + 2 + POLYTOPE_SIZE) {
            //     std::cout << "y: " << y.transpose() << std::endl;
            // }

            return y.squaredNorm();     // Assumes all the functions have the form of a norm
        }

    protected:
//        static void ConvertJacobianToVectorSum(matrixx_t& jac, vectorx_t& linear_term) {
//            jac.transposeInPlace();
//            linear_term.resize(jac.rows());
//            linear_term.setZero();
//
//            for (int col = 0; col < jac.cols(); col++) {
//                linear_term += jac.col(col);
//            }
//        }

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
                q_diff(i) = q_diff(i) * qref_qtarget_weight(2*config_size_ + i);
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
                v_diff(i) = v_diff(i) * vref_vtarget_weight(2*vel_size_ + i);
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
                tau_diff(i) = tau_diff(i) * tauref_tautarget_weight(2*input_size_ + i);
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
                force_diff(i) = force_diff(i) * fref_ftarget_weight(2 * force_size_ + i);
            }
        }

        void FootPolytopeCost(int frame_idx, const ad::ad_vector_t& dqk, const ad::ad_vector_t& qk_A_b_delta_mu_weight, ad::ad_vector_t& polytope_cost) const {
            const ad::ad_vector_t& qkbar = qk_A_b_delta_mu_weight.head(config_size_);
            ad::ad_matrix_t A = ad::ad_matrix_t::Zero(POLYTOPE_SIZE, 2);
            // ad::ad_matrix_t A = ad::ad_matrix_t::Identity(POLYTOPE_SIZE, 2);

            for (int i = 0; i < POLYTOPE_SIZE/2; i++) {
                A.row(i) = qk_A_b_delta_mu_weight.segment<2>(config_size_ + i*2).transpose();
            }

            A.bottomRows<POLYTOPE_SIZE/2>() = -A.topRows<POLYTOPE_SIZE/2>();

            ad::ad_vector_t b = qk_A_b_delta_mu_weight.segment<POLYTOPE_SIZE>(config_size_ + POLYTOPE_SIZE);        // ub then lb
            b.tail<POLYTOPE_SIZE/2>() = -b.tail<POLYTOPE_SIZE/2>();

            const ad::ad_vector_t q = torc::models::ConvertdqToq(dqk, qkbar);

            // Forward kinematics
            pinocchio::forwardKinematics(ad_pin_model_, *ad_pin_data_, q);
            pinocchio::updateFramePlacement(ad_pin_model_, *ad_pin_data_, frame_idx);

            // Get frame position in world frame (data oMf)
            ad::ad_vector_t frame_pos = ad_pin_data_->oMf.at(frame_idx).translation();

            polytope_cost.resize(POLYTOPE_SIZE);
            polytope_cost = A*frame_pos.head<2>() - b;
            // polytope_cost.setZero();

            // Now apply relaxed barrier
            const ad::adcg_t& delta = qk_A_b_delta_mu_weight(config_size_ + 2*POLYTOPE_SIZE);
            const ad::adcg_t& mu = qk_A_b_delta_mu_weight(config_size_ + 2*POLYTOPE_SIZE + 1);
            for (auto& val : polytope_cost) {
                ad::adcg_t relaxed_barrier = ((-val - 2*delta)/delta);
                // relaxed_barrier = relaxed_barrier*relaxed_barrier;
                relaxed_barrier = (relaxed_barrier*relaxed_barrier - 1)*mu/2 - mu*CppAD::log(delta);
                ad::adcg_t barrier = -mu*CppAD::log(-val);
                val = CppAD::CondExpGe(val, -delta, relaxed_barrier, 0*barrier);    // TODO: Why is the barrier value causing it to freak out?
                // val = -CppAD::log(-val + 1);
            }
        }

        /**
         * @brief Calculates the error frame location relative to a given location
         * @param dq
         * @param q_xyzdes_weight
         * @param frame_error
         */
        void FkCost(int frame_idx,
                    const torc::ad::ad_vector_t& dq,        // Change in configuration and velocity
                    const torc::ad::ad_vector_t& q_xyzdes_weight,    // configuration, desired position, weight
                    torc::ad::ad_vector_t& frame_error) {       // Error on the foot location
            const torc::ad::ad_vector_t& q = q_xyzdes_weight.head(config_size_);
            const Eigen::Vector3<torc::ad::adcg_t>& des_pos = q_xyzdes_weight.segment<3>(config_size_);

            // Get the current configuration
            const torc::ad::ad_vector_t q_curr = torc::models::ConvertdqToq(dq, q);

            // Get the frame location
            pinocchio::forwardKinematics(ad_pin_model_, *ad_pin_data_, q_curr);
            pinocchio::updateFramePlacement(ad_pin_model_, *ad_pin_data_, frame_idx);
            ad::ad_vector_t frame_pos = ad_pin_data_->oMf.at(frame_idx).translation();

            frame_error = frame_pos - des_pos;

            // Multiply by the weights
            for (int i = 0; i < POS_VARS; i++) {
                frame_error(i) = frame_error(i) * q_xyzdes_weight(config_size_ + POS_VARS + i);
            }
        }

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

        // TODO: Make this automatically match the variable in the MPC
        static constexpr int POLYTOPE_SIZE = 4;

        // Using explicit function here seemed to cause memory leaks
        // Also unclear if it is the memory leaks or the type here, but the MPC is notably quicker now
        std::map<std::string, std::unique_ptr<torc::ad::CppADInterface>> cost_fcn_terms_;

        std::vector<CostData> cost_data_;

        std::unique_ptr<CppAD::cg::DynamicLib<double>> config_jacobian_lib_;
        std::unique_ptr<CppAD::cg::GenericModel<double>> config_jacobian_model_;

        torc::models::ad_pin_model_t ad_pin_model_;
        std::shared_ptr<torc::models::ad_pin_data_t> ad_pin_data_;
    private:
    };
}    // namespace torc::mpc


#endif //COST_FUNCTION_H
