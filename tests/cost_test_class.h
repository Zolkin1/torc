//
// Created by zolkin on 8/3/24.
//

#ifndef COST_TEST_CLASS_H
#define COST_TEST_CLASS_H
#include "cost_function.h"
#include "full_order_rigid_body.h"
#include <catch2/catch_test_macros.hpp>

namespace torc::mpc {
    class CostTestClass : public CostFunction {
    public:
        explicit CostTestClass(const models::FullOrderRigidBody& model)
            : robot_model_(model) {}

        void CheckConfigure() {
            PrintTestHeader("Cost function configure");
            std::vector<vectorx_t> weights;
            weights.emplace_back(vectorx_t::Constant(robot_model_.GetConfigDim(), 1));
            weights.emplace_back(vectorx_t::Constant(robot_model_.GetVelDim(), 1));

            std::vector<CostTypes> costs;
            costs.emplace_back(Configuration);
            costs.emplace_back(Velocity);

            Configure(robot_model_.GetConfigDim(), robot_model_.GetVelDim(), robot_model_.GetNumInputs(), costs, weights);
            REQUIRE(configured_);
            REQUIRE(cost_idxs_.size() == costs.size());
            REQUIRE(cost_fcn_terms_.size() == costs.size());
            for (const auto& term : cost_fcn_terms_) {
                REQUIRE(term != nullptr);
            }
        }

        void CheckDerivatives() {
            PrintTestHeader("Cost function derivatives");
            for (int k = 0; k < 5; k++) {
                // ----- Configuration cost ----- //
                vectorx_t d_rand = robot_model_.GetRandomVel();
                vectorx_t bar_rand = robot_model_.GetRandomConfig();
                vectorx_t target_rand = robot_model_.GetRandomConfig();

                // Analytic
                vectorx_t arg;
                FormCostFcnArg(d_rand, bar_rand, target_rand, arg);
                vectorx_t grad_c = cost_fcn_terms_[cost_idxs_[Configuration]]->Gradient(arg);

                // Finite difference
                vectorx_t fd_c = cost_fcn_terms_[cost_idxs_[Configuration]]->GradientFiniteDiff(arg);

                CHECK(grad_c.isApprox(fd_c, sqrt(FD_DELTA)));

                // ----- Velocity cost ----- //
                vectorx_t vbar_rand = robot_model_.GetRandomVel();
                vectorx_t vtarget_rand = robot_model_.GetRandomVel();

                // Analytic
                FormCostFcnArg(d_rand, vbar_rand, vtarget_rand, arg);
                vectorx_t grad_v = cost_fcn_terms_[cost_idxs_[Velocity]]->Gradient(arg);

                // Finite difference
                vectorx_t fd_v = cost_fcn_terms_[cost_idxs_[Velocity]]->GradientFiniteDiff(arg);

                CHECK(grad_v.isApprox(fd_v, sqrt(FD_DELTA)));
            }
        }

        void CheckDefaultCosts() {
            PrintTestHeader("Cost function values");

            // ----- Configuration cost ----- //
            vectorx_t d = vectorx_t::Constant(robot_model_.GetVelDim(), 1);
            vectorx_t bar_rand = robot_model_.GetRandomConfig();
            vectorx_t target = bar_rand;

            double cost = GetTermCost(d, bar_rand, target, Configuration);
            CHECK(cost > 0);

            d.setZero();
            cost = GetTermCost(d, bar_rand, target, Configuration);
            CHECK(cost == 0);

            d.setRandom();
            cost = GetTermCost(d, bar_rand, target, Configuration);
            CHECK(cost >= 0);

            // ----- Veloctiy cost ----- //
            vectorx_t vbar_rand = robot_model_.GetRandomVel();
            vectorx_t vtarget = vbar_rand;

            d.setConstant(1);
            cost = GetTermCost(d, vbar_rand, vtarget, Velocity);
            CHECK(cost > 0);

            d.setZero();
            cost = GetTermCost(d, vbar_rand, vtarget, Velocity);
            CHECK(cost == 0);

            d.setRandom();
            cost = GetTermCost(d, vbar_rand, vtarget, Velocity);
            CHECK(cost >= 0);
        }

        // ---------------------- //
        // ----- Benchmarks ----- //
        // ---------------------- //
        void BenchmarkCostFunctions() {
            // ----- Configuration cost ----- //
            vectorx_t d_rand = robot_model_.GetRandomVel();
            vectorx_t bar_rand = robot_model_.GetRandomConfig();
            vectorx_t target_rand = robot_model_.GetRandomConfig();

            // Analytic
            vectorx_t arg;
            FormCostFcnArg(d_rand, bar_rand, target_rand, arg);
            BENCHMARK("configuration cost function gradient") {
                vectorx_t grad_c = cost_fcn_terms_[cost_idxs_[Configuration]]->Gradient(arg);
            };

            BENCHMARK("configuration cost function evaluation") {
                double c1 = cost_fcn_terms_[cost_idxs_[Configuration]]->Evaluate(arg);
            };

            // ----- Velocity cost ----- //
            vectorx_t vbar_rand = robot_model_.GetRandomVel();
            vectorx_t vtarget_rand = robot_model_.GetRandomVel();

            // Analytic
            FormCostFcnArg(d_rand, vbar_rand, vtarget_rand, arg);
            BENCHMARK("velocity cost function gradient") {
                vectorx_t grad_v = cost_fcn_terms_[cost_idxs_[Velocity]]->Gradient(arg);
            };

            BENCHMARK("configuration cost function evaluation") {
                double c1 = cost_fcn_terms_[cost_idxs_[Velocity]]->Evaluate(arg);
            };

        }
    protected:
        models::FullOrderRigidBody robot_model_;
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
