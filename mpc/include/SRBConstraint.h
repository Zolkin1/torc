//
// Created by zolkin on 1/28/25.
//

#ifndef SRBDYNAMICS_H
#define SRBDYNAMICS_H

#include <filesystem>

#include "full_order_rigid_body.h"
#include "constraint.h"
#include "MpcSettings.h"


namespace torc::mpc {
    class SRBConstraint : public Constraint {
    public:
        SRBConstraint(int first_node, int last_node, const std::string& name, const std::vector<std::string>& contact_frames,
            const std::filesystem::path& deriv_lib_path, bool compile_derivs, const models::FullOrderRigidBody& model,
            const vectorx_t& q_nom);

        void GetLinDynamics(const vectorx_t& q1_lin, const vectorx_t& q2_lin,
            const vectorx_t& v1_lin, const vectorx_t& v2_lin, const vectorx_t& force_lin,
            double dt, matrixx_t& A, matrixx_t& B, vectorx_t& b);

        int GetNumConstraints() const override;

        std::pair<vectorx_t, vectorx_t> GetViolation(const vectorx_t& q1_lin, const vectorx_t& q2_lin,
            const vectorx_t& v1_lin, const vectorx_t& v2_lin, const vectorx_t& force_lin,
            double dt, const vectorx_t& dq1, const vectorx_t& dq2,
            const vectorx_t& dv1, const vectorx_t& dv2, const vectorx_t& dforce);
    protected:
    private:
        void IntegrationConstraint(const ad::ad_vector_t& dqk_dqkp1_dvk,
            const ad::ad_vector_t& dt_qkbar_qkp1bar_vk_vkp1, ad::ad_vector_t& violation);


        void SRBDynamics(const std::vector<std::string> &frames,
            const ad::ad_vector_t &dqk_dvk_dvkp1_dfk, const ad::ad_vector_t &qk_vk_vkp1_fk_dt,
            ad::ad_vector_t &violation);

        int vel_dim_;
        int config_dim_;
        int tau_dim_;
        int num_contacts_;

        pinocchio::Model srb_pin_model_;
        models::ad_pin_model_t ad_srb_pin_model_;

        std::unique_ptr<pinocchio::Data> srb_pin_data_;
        std::shared_ptr<models::ad_pin_data_t> ad_srb_pin_data_;

        // models::FullOrderRigidBody srb_model_;
        models::FullOrderRigidBody model_;
        std::unique_ptr<ad::CppADInterface> dynamics_function_;
        std::unique_ptr<ad::CppADInterface> integration_function_;
        std::vector<std::string> contact_frames_;

        vector3_t base_to_com_;
        matrix3_t base_to_com_skew_sym_;
        matrix3_t srb_intertia_;
        matrix3_t srb_intertia_inv_;

        static constexpr int CONTACT_3DOF = 3;
        static constexpr int POS_VARS = 3;
        static constexpr int QUAT_VARS = 4;
        static constexpr int FLOATING_BASE = 7;
        static constexpr int FLOATING_VEL = 6;
    };
}


#endif //SRBDYNAMICS_H
