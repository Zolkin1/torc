//
// Created by zolkin on 1/18/25.
//

#ifndef STATECONSTRAINT_H
#define STATECONSTRAINT_H
#include "constraint.h"
#include "full_order_rigid_body.h"

namespace torc::mpc {
    class SwingConstraint : public Constraint {
    public:
        SwingConstraint(int first_node, int last_node, const std::string& name, const models::FullOrderRigidBody& model,
            const std::vector<std::string>& contact_frames,
            const std::filesystem::path& deriv_lib_path, bool compile_derivs);

        std::pair<matrixx_t, vectorx_t> GetLinearization(const vectorx_t& q_lin, double des_height, const std::string& frame);

    protected:
    private:
        void SwingHeightConstraint(const std::string& frame, const ad::ad_vector_t& dqk,
            const ad::ad_vector_t& qk_desheight, ad::ad_vector_t& violation);

        std::vector<std::string> contact_frames_;
        models::FullOrderRigidBody model_;
        std::map<std::string, std::unique_ptr<ad::CppADInterface>> swing_function_;
    };
}


#endif //STATECONSTRAINT_H
