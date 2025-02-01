//
// Created by zolkin on 2/1/25.
//

#ifndef CENTROIDALDYNAMICSCONSTRAINT_H
#define CENTROIDALDYNAMICSCONSTRAINT_H

#include "full_order_rigid_body.h"

#include "constraint.h"

namespace torc::mpc {
    class CentroidalDynamicsConstraint : public Constraint {
    public:
        CentroidalDynamicsConstraint(const models::FullOrderRigidBody& model,
            const std::vector<std::string>& contact_frames, const std::string& name,
            const std::filesystem::path& deriv_lib_path, bool compile_derivs,
            int first_node, int last_node);

        void GetLinDynamics(const vectorx_t& q1_lin, const vectorx_t& q2_lin,
            const vectorx_t& v1_lin, const vectorx_t& v2_lin, const vectorx_t& force_lin,
            double dt, matrixx_t& A, matrixx_t& B, vectorx_t& b);

        int GetNumConstraints() const override;

        // bool IsInNodeRange(int node) const override;

        std::pair<vectorx_t, vectorx_t> GetViolation(const vectorx_t& q1_lin, const vectorx_t& q2_lin,
            const vectorx_t& v1_lin, const vectorx_t& v2_lin, const vectorx_t& force_lin,
            double dt, const vectorx_t& dq1, const vectorx_t& dq2,
            const vectorx_t& dv1, const vectorx_t& dv2, const vectorx_t& dforce);
    protected:

        void CentroidalInverseDynamics(const std::vector<std::string> &frames,
            const ad::ad_vector_t &dqk_dvk_dvkp1base_dfk, const ad::ad_vector_t &qk_vk_vkp1base_fk_dt,
            ad::ad_vector_t &violation);

        void IntegrationConstraint(const ad::ad_vector_t& dqk_dqkp1_dvk,
            const ad::ad_vector_t& dt_qkbar_qkp1bar_vk, ad::ad_vector_t& violation);

        int vel_dim_;
        int config_dim_;
        int num_contacts_;
        std::vector<std::string> contact_frames_;

        models::FullOrderRigidBody model_;
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


#endif //CENTROIDALDYNAMICSCONSTRAINT_H
