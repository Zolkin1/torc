//
// Created by zolkin on 8/3/24.
//

#ifndef COST_TEST_CLASS_H
#define COST_TEST_CLASS_H
#include "cost_function.h"
#include "full_order_rigid_body.h"
#include <catch2/catch_test_macros.hpp>

namespace torc::mpc {
    // TODO: Update for new costs!
    class CostTestClass : public CostFunction {
    public:
        explicit CostTestClass(const std::string& name, const fs::path& urdf_path)
            : CostFunction("test_cost") {
            robot_model_ = std::make_unique<models::FullOrderRigidBody>(name, urdf_path);
        }

        void CheckConfigure() {
            PrintTestHeader("Cost function configure");
            std::vector<vectorx_t> weights;
            weights.emplace_back(vectorx_t::Constant(robot_model_->GetVelDim(), 1));
            weights.emplace_back(vectorx_t::Constant(robot_model_->GetVelDim(), 1));
            weights.emplace_back(vectorx_t::Constant(robot_model_->GetNumInputs(), 1));

            std::vector<CostTypes> costs;
            costs.emplace_back(Configuration);
            costs.emplace_back(VelocityTracking);
            costs.emplace_back(TorqueReg);

            Configure(robot_model_, true, costs, weights, std::filesystem::current_path()/"cost_test_deriv_libs");
            REQUIRE(configured_);
            REQUIRE(cost_idxs_.size() == costs.size());
            REQUIRE(cost_fcn_terms_.size() == costs.size());
            for (const auto& term : cost_fcn_terms_) {
                REQUIRE(term != nullptr);
            }
        }

        void CheckDerivatives() {
            PrintTestHeader("Cost function derivatives");
            double constexpr FD_DELTA = 1e-8;

            for (int k = 0; k < 5; k++) {
                // ----- Configuration cost ----- //
                vectorx_t d_rand = robot_model_->GetRandomVel();
                vectorx_t d_zero =  vectorx_t::Zero(robot_model_->GetVelDim());
                vectorx_t bar_rand = robot_model_->GetRandomConfig();
                vectorx_t target_rand = robot_model_->GetRandomConfig();

                // Analytic
                vectorx_t grad_c;
                matrixx_t hess_c;
                GetApproximation(bar_rand, target_rand, grad_c, hess_c, CostTypes::Configuration);

                // Finite difference
                vectorx_t fd_grad_c(grad_c.size());
                double val = GetTermCost(d_zero, bar_rand, target_rand, CostTypes::Configuration);
                for (int i = 0; i < d_rand.size(); i++) {
                    d_zero(i) += FD_DELTA;
                    fd_grad_c(i) = (GetTermCost(d_zero, bar_rand, target_rand, CostTypes::Configuration) - val)/FD_DELTA;
                    d_zero(i) -= FD_DELTA;
                }

                CHECK(grad_c.isApprox(fd_grad_c, sqrt(FD_DELTA)));

                // ----- Velocity cost ----- //
                vectorx_t vbar_rand = robot_model_->GetRandomVel();
                vectorx_t vtarget_rand = robot_model_->GetRandomVel();

                // Analytic
                vectorx_t grad_v;
                matrixx_t hess_v;
                GetApproximation(vbar_rand, vtarget_rand, grad_v, hess_v, CostTypes::VelocityTracking);

                // Finite difference
                vectorx_t fd_grad_v(grad_v.size());
                val = GetTermCost(d_zero, vbar_rand, vtarget_rand, CostTypes::VelocityTracking);
                for (int i = 0; i < d_rand.size(); i++) {
                    d_zero(i) += FD_DELTA;
                    fd_grad_v(i) = (GetTermCost(d_zero, vbar_rand, vtarget_rand, CostTypes::VelocityTracking) - val)/FD_DELTA;
                    d_zero(i) -= FD_DELTA;
                }

                CHECK(grad_v.isApprox(fd_grad_v, sqrt(FD_DELTA)));

                // Hessian
                matrixx_t fd_hess_v(d_rand.size(), d_rand.size());
                vectorx_t grad_v_temp(d_rand.size());
                for (int i = 0; i < d_rand.size(); i++) {
                    matrixx_t jac_temp;
                    vectorx_t p(3*robot_model_->GetVelDim());
                    p << vbar_rand, vtarget_rand, weights_[cost_idxs_[CostTypes::VelocityTracking]];
                    d_zero(i) += FD_DELTA;
                    cost_fcn_terms_[cost_idxs_[CostTypes::VelocityTracking]]->GetJacobian(d_zero, p, jac_temp);
                    vectorx_t y;
                    cost_fcn_terms_[cost_idxs_[CostTypes::VelocityTracking]]->GetFunctionValue(d_zero, p, y);
                    grad_v_temp= 2*jac_temp.transpose()*y;

                    fd_hess_v.col(i) = (grad_v_temp - grad_v)/FD_DELTA;

                    d_zero(i) -= FD_DELTA;
                }

//                std::cout << "analytic hess: \n" << hess_v << std::endl;
//                std::cout << "fd hess: \n" << fd_hess_v << std::endl;
                CHECK(hess_v.isApprox(fd_hess_v, sqrt(FD_DELTA)));


                // ----- Check values with approximations ----- //
                // Velocity tracking, force reg and torque reg should be exact
                vectorx_t dv_rand = robot_model_->GetRandomVel();
                double vel_error = GetTermCost(dv_rand, vbar_rand, vtarget_rand, CostTypes::VelocityTracking);

                GetApproximation(vbar_rand, vtarget_rand, grad_v, hess_v, CostTypes::VelocityTracking);
                double vel_error_approx = 0.5*dv_rand.transpose()*hess_v*dv_rand + grad_v.dot(dv_rand) +
                        GetTermCost(vectorx_t::Zero(robot_model_->GetVelDim()), vbar_rand, vtarget_rand, CostTypes::VelocityTracking);

                CHECK(std::abs(vel_error - vel_error_approx) < 1e-8);

            }
        }

//        void CheckLinearizeQuadrasize() {
//            PrintTestHeader("Linearization and Quadratasize");
//            // ------------------------- //
//            // ----- Linearization ----- //
//            // ------------------------- //
//
//            // ----- Configuration cost ----- //
//            vectorx_t d = vectorx_t::Zero(robot_model_.GetVelDim());
//            vectorx_t bar_rand = robot_model_.GetRandomConfig();
//            vectorx_t target_rand = robot_model_.GetRandomConfig();
//            vectorx_t lin_term = vectorx_t::Zero(robot_model_.GetVelDim());
//            Linearize(bar_rand, target_rand, Configuration, lin_term);
//
//            // Finite difference
//            vectorx_t arg;
//            FormCostFcnArg(d, bar_rand, target_rand, arg);
//            vectorx_t fd_c = cost_fcn_terms_[cost_idxs_[Configuration]]->GradientFiniteDiff(arg);
//            vectorx_t partial_bar = fd_c.head(robot_model_.GetVelDim());
//
//            REQUIRE(lin_term.size() == partial_bar.size());
//            CHECK(lin_term.isApprox(partial_bar, sqrt(FD_DELTA)));
//
//            // ----- Velocity cost ----- //
//            vectorx_t vbar_rand = robot_model_.GetRandomVel();
//            vectorx_t vtarget_rand = robot_model_.GetRandomVel();
//            Linearize(vbar_rand, vtarget_rand, VelocityTracking, lin_term);
//
//            // Finite difference
//            FormCostFcnArg(d, vbar_rand, vtarget_rand, arg);
//            vectorx_t fd_v = cost_fcn_terms_[cost_idxs_[VelocityTracking]]->GradientFiniteDiff(arg);
//            vectorx_t partial_bar_v = fd_v.head(robot_model_.GetVelDim());
//
//            REQUIRE(lin_term.size() == partial_bar_v.size());
//            CHECK(lin_term.isApprox(partial_bar_v, sqrt(FD_DELTA)));
//
//            // ------------------------- //
//            // -------- Jacobian ------- //
//            // ------------------------- //
//            FormCostFcnArg(d, bar_rand, target_rand, arg);
//            matrixx_t jac = GetConfigurationTrackingJacobian(arg);
//            matrixx_t gtg_from_jac = 4*(jac*GetConfigurationDiffVector(arg))*(jac*GetConfigurationDiffVector(arg)).transpose();
//            Linearize(bar_rand, target_rand, Configuration, lin_term);
//
//            matrixx_t gtg = lin_term*lin_term.transpose();
//
//            CHECK(gtg_from_jac.isApprox(gtg));
//
//            // ------------------------- //
//            // ------- Quadratic ------- //
//            // ------------------------- //
//            matrixx_t hess_term;
//            Quadraticize(bar_rand, target_rand, Configuration, hess_term);
//            REQUIRE(hess_term.rows() == robot_model_.GetVelDim());
//            REQUIRE(hess_term.cols() == robot_model_.GetVelDim());
//
//            // FormCostFcnArg(d, bar_rand, target_rand, arg);
//            // matrixx_t fd_c = cost_fcn_terms_[cost_idxs_[Configuration]]->GradientFiniteDiff(arg);
//            // vectorx_t partial_bar = fd_c.head(robot_model_.GetVelDim());
//
//        }

        void CheckDefaultCosts() {
            PrintTestHeader("Cost function values");
            // std::cout << "weight config: " << weights_[0].transpose() << std::endl;

            // ----- Configuration cost ----- //
            vectorx_t d = vectorx_t::Constant(robot_model_->GetVelDim(), 1);
            vectorx_t bar_rand = robot_model_->GetRandomConfig();
            vectorx_t target = bar_rand;

            double cost = GetTermCost(d, bar_rand, target, Configuration);
            CHECK(cost > 0);

            d.setZero();
            cost = GetTermCost(d, bar_rand, target, Configuration);
            CHECK(std::abs(cost) < 1e-8);

            d.setRandom();
            cost = GetTermCost(d, bar_rand, target, Configuration);
            CHECK(cost >= 0);

            // ----- Veloctiy cost ----- //
            vectorx_t vbar_rand = robot_model_->GetRandomVel();
            vectorx_t vtarget = vbar_rand;

            d.setConstant(1);
            cost = GetTermCost(d, vbar_rand, vtarget, VelocityTracking);
            CHECK(cost > 0);

            d.setZero();
            cost = GetTermCost(d, vbar_rand, vtarget, VelocityTracking);
            CHECK(cost == 0);

            d.setRandom();
            cost = GetTermCost(d, vbar_rand, vtarget, VelocityTracking);
            CHECK(cost >= 0);
        }
//
//        // ---------------------- //
//        // ----- Benchmarks ----- //
//        // ---------------------- //
//        void BenchmarkCostFunctions() {
//            // ----- Configuration cost ----- //
//            vectorx_t d_rand = robot_model_.GetRandomVel();
//            vectorx_t bar_rand = robot_model_.GetRandomConfig();
//            vectorx_t target_rand = robot_model_.GetRandomConfig();
//
//            // Analytic
//            vectorx_t arg;
//            FormCostFcnArg(d_rand, bar_rand, target_rand, arg);
//            BENCHMARK("configuration cost function gradient") {
//                vectorx_t grad_c = cost_fcn_terms_[cost_idxs_[Configuration]]->Gradient(arg);
//            };
//
//            BENCHMARK("configuration cost function evaluation") {
//                double c1 = cost_fcn_terms_[cost_idxs_[Configuration]]->Evaluate(arg);
//            };
//
//            // ----- Velocity cost ----- //
//            vectorx_t vbar_rand = robot_model_.GetRandomVel();
//            vectorx_t vtarget_rand = robot_model_.GetRandomVel();
//
//            // Analytic
//            FormCostFcnArg(d_rand, vbar_rand, vtarget_rand, arg);
//            BENCHMARK("velocity cost function gradient") {
//                vectorx_t grad_v = cost_fcn_terms_[cost_idxs_[VelocityTracking]]->Gradient(arg);
//            };
//
//            BENCHMARK("configuration cost function evaluation") {
//                double c1 = cost_fcn_terms_[cost_idxs_[VelocityTracking]]->Evaluate(arg);
//            };
//
//        }
    protected:
        std::unique_ptr<models::FullOrderRigidBody> robot_model_;
    private:
        static constexpr double FD_DELTA = 1e-8;

        void PrintTestHeader(const std::string& name) {
            using std::setw;
            using std::setfill;

            const int total_width = 50;
            std::cout << setfill('=') << setw(total_width/2 - name.size()/2) << " " << name << " " << setw(total_width/2 - name.size()/2) << "" << std::endl;
        }
    };
}

#endif //COST_TEST_CLASS_H
