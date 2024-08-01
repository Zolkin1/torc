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

        void BenchmarkQuaternionIntegrationLin() {
            BENCHMARK("quaternion integration lin") {
                auto deriv = GetQuatIntegrationLinearizationXi(0);
            };
        }

        void BenchmarkQuaternionConfigurationLin() {
            BENCHMARK("quaternion configuration lin") {
                auto deriv = GetQuatLinearization(0);
            };
        }
    protected:
    private:
    };
} // namespacre torc::mpc

#endif //MPC_TEST_CLASS_H
