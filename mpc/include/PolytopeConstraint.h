//
// Created by zolkin on 1/30/25.
//

#ifndef POLYTOPECONSTRAINT_H
#define POLYTOPECONSTRAINT_H
#include "constraint.h"
#include "full_order_rigid_body.h"
#include "contact_schedule.h"

namespace torc::mpc {
    class PolytopeConstraint : public Constraint {
    public:
        PolytopeConstraint(int first_node, int last_node, const std::string& name,
        const std::vector<std::string>& polytope_frames,
        const std::filesystem::path& deriv_lib_path, bool compile_derivs,
        const models::FullOrderRigidBody& model);

        void GetLinearization(const vectorx_t& q_lin, const ContactInfo& contact_info,
            const std::string& frame, matrixx_t& lin_mat, vectorx_t& ub, vectorx_t& lb);

        vectorx_t GetViolation(const vectorx_t& q_lin, const vectorx_t& dq,
            const ContactInfo& contact_info, const std::string& frame);

        int GetNumConstraints() const override;

        static constexpr int POLYTOPE_SIZE = 4;

        std::vector<std::string> GetPolytopeFrames() const;

    protected:
    private:
        void FootPolytopeConstraint(const std::string& frame, const ad::ad_vector_t& dqk,
            const ad::ad_vector_t& qk_A, ad::ad_vector_t& violation);

        std::map<std::string, std::unique_ptr<ad::CppADInterface>> constraint_functions_;
        models::FullOrderRigidBody model_;

        int config_dim_;

        std::vector<std::string> polytope_frames_;
    };
}


#endif //POLYTOPECONSTRAINT_H
