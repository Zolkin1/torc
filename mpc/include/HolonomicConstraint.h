//
// Created by zolkin on 1/19/25.
//

#ifndef HOLONOMICCONSTRAINT_H
#define HOLONOMICCONSTRAINT_H
#include "constraint.h"
#include "full_order_rigid_body.h"


namespace torc::mpc {
    class HolonomicConstraint : public Constraint {
    public:
        HolonomicConstraint(int first_node, int last_node, const std::string& name, const models::FullOrderRigidBody& model,
            const std::vector<std::string>& contact_frames,
            const std::filesystem::path& deriv_lib_path, bool compile_derivs);

        std::pair<matrixx_t, vectorx_t> GetLinearization(const vectorx_t& q_lin, const vectorx_t& v_lin,
            const std::string& frame);
    protected:
    private:
        void HoloConstraint(const std::string& frame, const ad::ad_vector_t& dqk_dvk, const ad::ad_vector_t& qk_vk, ad::ad_vector_t& violation);

        models::FullOrderRigidBody model_;
        std::map<std::string, std::unique_ptr<ad::CppADInterface>> constraint_functions_;
    };
}



#endif //HOLONOMICCONSTRAINT_H
