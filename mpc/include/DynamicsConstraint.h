//
// Created by zolkin on 1/18/25.
//

#ifndef DYNAMICSCONSTRAINT_H
#define DYNAMICSCONSTRAINT_H
#include "constraint.h"
#include "full_order_rigid_body.h"
#include <filesystem>

namespace torc::mpc {
    namespace fs = std::filesystem;
    class DynamicsConstraint : public Constraint {
    public:
        DynamicsConstraint(const models::FullOrderRigidBody& model,
            const std::vector<std::string>& contact_frames, const std::string& name, const fs::path& deriv_lib_path,
            bool compile_derivs,
            int first_node, int last_node);

        void GetLinDynamics(const vectorx_t& q1_lin, const vectorx_t& q2_lin,
            const vectorx_t& v1_lin, const vectorx_t& v2_lin, const vectorx_t& tau_lin, const vectorx_t& force_lin,
            double dt, bool boundary, matrixx_t& A, matrixx_t& B, vectorx_t& b);

        int GetNumConstraints() const override;

        std::pair<vectorx_t, vectorx_t> GetViolation(const vectorx_t& q1_lin, const vectorx_t& q2_lin,
            const vectorx_t& v1_lin, const vectorx_t& v2_lin, const vectorx_t& tau_lin, const vectorx_t& force_lin,
            double dt, const vectorx_t& dq1, const vectorx_t& dq2,
            const vectorx_t& dv1, const vectorx_t& dv2, const vectorx_t& dtau, const vectorx_t& dforce);

    protected:
        void InverseDynamics(const std::vector<std::string> &frames,
            const ad::ad_vector_t &dqk_dvk_dvkp1_dtauk_dfk, const ad::ad_vector_t &qk_vk_vkp1_tauk_fk_dt,
            ad::ad_vector_t &violation);

        void IntegrationConstraint(const ad::ad_vector_t& dqk_dqkp1_dvk,
            const ad::ad_vector_t& dt_qkbar_qkp1bar_vk_vkp1, ad::ad_vector_t& violation);

        void ComputeDynamicsJacobians(const vectorx_t& q1_lin,
            const vectorx_t& v1_lin, const vectorx_t& v2_lin, const vectorx_t& tau_lin, const vectorx_t& force_lin,
            double dt, const vectorx_t& dq1, const vectorx_t& dv1, const vectorx_t& dtau,
            const vectorx_t& dforce, matrixx_t& Jdq, matrixx_t& Jdv, matrixx_t& Jdtau, matrixx_t& JdF, vectorx_t& b);

        int vel_dim_;
        int config_dim_;
        int tau_dim_;
        int num_contacts_;
        std::vector<std::string> contact_frames_;

        models::FullOrderRigidBody model_;
        pinocchio::Data pin_data_;
        std::unique_ptr<ad::CppADInterface> dynamics_function_;
        std::unique_ptr<ad::CppADInterface> integration_function_;

        static constexpr int CONTACT_3DOF = 3;
        static constexpr int POS_VARS = 3;
        static constexpr int QUAT_VARS = 4;
        static constexpr int FLOATING_BASE = 7;
        static constexpr int FLOATING_VEL = 6;

    private:
    };
}


#endif //DYNAMICSCONSTRAINT_H
