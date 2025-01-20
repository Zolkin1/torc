//
// Created by zolkin on 1/19/25.
//

#ifndef DYNAMICSCONSTRAINTSTEST_H
#define DYNAMICSCONSTRAINTSTEST_H
#include "DynamicsConstraint.h"


namespace torc::mpc {
    class DynamicsConstraintsTest : public DynamicsConstraint {
    public:
        DynamicsConstraintsTest(const models::FullOrderRigidBody& model,
            const std::vector<std::string>& contact_frames, const std::string& name, const fs::path& deriv_lib_path,
            bool compile_derivs, bool full_order,
            int first_node, int last_node);

        void FiniteDiffForwardDynamics(const vectorx_t& q_lin, const vectorx_t& v1_lin,
            const vectorx_t& v2_lin, const vectorx_t& tau_lin, const vectorx_t& F_lin, double dt);

        bool ForwardDynamicsTest(const vectorx_t& q_lin, const vectorx_t& v1_lin, const vectorx_t& v2_lin,
            const vectorx_t& tau_lin, const vectorx_t& F_lin);

        std::vector<std::string> contact_frames_;
    };
}

#endif //DYNAMICSCONSTRAINTSTEST_H
