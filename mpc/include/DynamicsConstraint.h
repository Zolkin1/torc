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
            bool compile_derivs, bool full_order,
            int first_node, int last_node);

        std::pair<matrixx_t, matrixx_t> GetLinDynamics(const vectorx_t& q1_lin, const vectorx_t& q2_lin,
            const vectorx_t& v1_lin, const vectorx_t& v2_lin, const vectorx_t& tau_lin, const vectorx_t& force_lin,
            double dt);

        int GetNumConstraints() const override;

        // These dynamics convert the full order model to the centroidal ones.
        std::pair<matrixx_t, matrixx_t> GetBoundaryDynamics();

        bool IsInNodeRange(int node) const override;

    protected:
        // Forward dynamics constraint
        // void ForwardDynamics(const std::vector<std::string>& frames,
        // const ad::ad_vector_t& dqk_dvk_dvkp1_dtauk_dfk, const ad::ad_vector_t& qk_vk_vkp1_tauk_fk_dt, ad::ad_vector_t& violation);
        void InverseDynamics(const std::vector<std::string> &frames,
            const ad::ad_vector_t &dqk_dvk_dvkp1_dtauk_dfk, const ad::ad_vector_t &qk_vk_vkp1_tauk_fk_dt,
            ad::ad_vector_t &violation);

        void IntegrationConstraint(const ad::ad_vector_t& dqk_dqkp1_dvk_dvkp1,
            const ad::ad_vector_t& dt_qkbar_qkp1bar_vk_vkp1, ad::ad_vector_t& violation);

        int nx_;
        int nu_;
        int vel_dim_;
        int config_dim_;
        int tau_dim_;
        int num_contacts_;

        bool full_order_;

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


#endif //DYNAMICSCONSTRAINT_H
